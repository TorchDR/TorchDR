"""Real-process tests for exact k-NN over a database sharded across ranks.

Ordinary CI runs these against Gloo and CPU FAISS. The same module can run with
NCCL and GPU FAISS, because TorchDR selects the process-group backend from the
available device. Each rank's result must equal replicated Flat search for its
query chunk.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import torchdr.distance.faiss as faiss_mod
from torchdr.distance import FaissConfig, FaissPlanConfig, pairwise_distances
from torchdr.distance.faiss import sharded_pairwise_distances_faiss
from torchdr.distributed import (
    DistributedContext,
    init_distributed,
    shutdown_distributed,
)
from torchdr.utils import faiss


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
        reason="run through the dedicated multi-process integration workflow",
    ),
    pytest.mark.skipif(faiss is None or faiss is False, reason="faiss not installed"),
]

N_SAMPLES = 4000
N_FEATURES = 8
K = 5


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the process group created by torchrun.

    Fails loudly on a single process: with one rank the whole database is the
    only shard and the merge never crosses a rank boundary, so a silent one-rank
    run would report green while exercising none of the sharded search.
    """
    init_distributed()
    world_size = dist.get_world_size()
    if world_size < 2:
        shutdown_distributed()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    shutdown_distributed()


@pytest.fixture(scope="module")
def context(distributed_process_group):
    return DistributedContext()


@pytest.fixture(scope="module")
def data():
    """The same dataset on every rank, which is the contract TorchDR expects."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N_SAMPLES, N_FEATURES, generator=generator)


def _dataset(n_samples, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, N_FEATURES, generator=generator)


def _reference(data, k, metric, exclude_diag=False):
    """Single-process exact Flat neighbors over the full database."""
    return pairwise_distances(
        data,
        k=k,
        metric=metric,
        backend=FaissConfig(),
        exclude_diag=exclude_diag,
        return_indices=True,
    )


def _sharded(data, k, metric, context, exclude_diag=False):
    """Sharded neighbors for this rank's chunk, through the plan API."""
    return pairwise_distances(
        data,
        k=k,
        metric=metric,
        backend=FaissPlanConfig(distribution="shard"),
        exclude_diag=exclude_diag,
        return_indices=True,
        distributed_ctx=context,
    )


def _assert_matches_chunk(data, context, k, metric, exclude_diag=False):
    ref_D, ref_I = _reference(data, k, metric, exclude_diag)
    sh_D, sh_I = _sharded(data, k, metric, context, exclude_diag)
    start, end = context.compute_chunk_bounds(data.shape[0])
    assert sh_I.shape == (end - start, k)
    assert torch.equal(sh_I, ref_I[start:end])
    assert torch.allclose(sh_D, ref_D[start:end])


class TestShardedSearch:
    @pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean", "angular"])
    def test_shard_matches_single_process_flat(self, data, context, metric):
        # The heart of #301: sharding the database and merging the per-shard
        # candidates must reproduce the single-process neighbors exactly, for
        # both the L2 metrics and the inner-product (angular) ordering.
        _assert_matches_chunk(data, context, K, metric)

    def test_shard_excludes_self_neighbor(self, data, context):
        # exclude_diag removes each query's own row, matched by global index so
        # it is correct even though the query sits in a different rank's shard.
        _assert_matches_chunk(data, context, K, "sqeuclidean", exclude_diag=True)

    def test_shard_handles_uneven_partitions(self, context):
        # A sample count divisible by neither two nor four gives ranks chunks of
        # different sizes; the merge must still line up with the global result.
        uneven = _dataset(4003)
        _assert_matches_chunk(uneven, context, K, "sqeuclidean")
        _assert_matches_chunk(uneven, context, K, "angular")

    def test_shard_handles_k_larger_than_a_shard(self, context):
        # With few samples each shard holds fewer than k rows, so every local
        # search returns padding (-1) that the merge must discard while still
        # assembling the correct k global neighbors.
        small = _dataset(9)
        _assert_matches_chunk(small, context, 5, "sqeuclidean")
        _assert_matches_chunk(small, context, 5, "sqeuclidean", exclude_diag=True)

    def test_bounded_query_batch_matches(self, data, context):
        # A tiny query batch forces many broadcast/all_gather rounds; the neighbor
        # identities must be identical, which is what keeps the merge memory bounded
        # by the batch rather than the dataset. FAISS computes ``||q-p||^2`` with a
        # blocked matmul whose last-bit rounding depends on the query-batch size, so
        # the indices are exact while the raw distances match only up to that
        # ~1e-6 wobble.
        ref_D, ref_I = _reference(data, K, "sqeuclidean")
        sh_D, sh_I = sharded_pairwise_distances_faiss(
            data,
            k=K,
            metric="sqeuclidean",
            distributed_ctx=context,
            query_batch_size=7,
        )
        start, end = context.compute_chunk_bounds(N_SAMPLES)
        assert torch.equal(sh_I, ref_I[start:end])
        assert torch.allclose(sh_D, ref_D[start:end], rtol=1e-4, atol=1e-4)

    def test_collectives_stay_in_lockstep_across_ranks(self, context):
        # #301 requires the sharded collectives never deadlock. The loop walks a
        # fixed source order 0..world_size-1 with a shared batch size, so every
        # rank must issue the identical broadcast/all_gather sequence and no rank
        # can wait on a collective a peer never posts. Uneven partitions (a count
        # divisible by neither two nor four) are exactly where a per-rank shortcut
        # would desynchronize the counts, so they are the sharpest guard.
        uneven = _dataset(4003)
        with (
            mock.patch.object(
                faiss_mod.dist, "broadcast", side_effect=faiss_mod.dist.broadcast
            ) as bcast,
            mock.patch.object(
                faiss_mod.dist, "all_gather", side_effect=faiss_mod.dist.all_gather
            ) as gather,
        ):
            sharded_pairwise_distances_faiss(
                uneven,
                k=K,
                metric="sqeuclidean",
                distributed_ctx=context,
                query_batch_size=500,
            )
            local = torch.tensor(
                [bcast.call_count, gather.call_count], dtype=torch.long
            )
        # Distances and indices are gathered once each per broadcast round, and
        # every source rank contributes at least one round.
        assert local[1].item() == 2 * local[0].item()
        assert local[0].item() >= context.world_size
        # The real all_gather (restored on exit) confirms the counts agree; a
        # mismatch here is the signature of a latent deadlock.
        gathered = [torch.empty_like(local) for _ in range(context.world_size)]
        dist.all_gather(gathered, local)
        per_rank = [tuple(t.tolist()) for t in gathered]
        assert len(set(per_rank)) == 1, (
            f"ranks issued different collective counts and would deadlock: {per_rank}"
        )
