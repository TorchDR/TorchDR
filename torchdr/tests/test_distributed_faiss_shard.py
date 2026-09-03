"""Real-process tests for exact k-NN over a database sharded across ranks.

These run on the Gloo CPU backend against a CPU FAISS index, so ordinary CI
runners cover the broadcast/all_gather merge that lets each rank index only its
own shard yet still return the exact global neighbors for its chunk. The
contract mirrors the replicated path: every rank's result equals the
single-process Flat search restricted to that rank's chunk.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr.distance import FaissConfig, FaissPlanConfig, pairwise_distances
from torchdr.distance.faiss import sharded_pairwise_distances_faiss
from torchdr.distributed import DistributedContext
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
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    if world_size < 2:
        dist.destroy_process_group()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    dist.destroy_process_group()


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


def _assert_neighbor_set_matches_chunk(data, context, k, metric):
    """Compare neighbor sets and distances without requiring a tie-break order.

    When ``k`` reaches deep into the ranking, equal-distance neighbors appear and
    neither FAISS nor the merge defines their relative order. The exact result is
    then the neighbor *set* and the sorted distances; any index that differs
    between the two must sit at an exactly tied distance, so a genuinely wrong
    neighbor still fails this check.
    """
    ref_D, ref_I = _reference(data, k, metric)
    sh_D, sh_I = _sharded(data, k, metric, context)
    start, end = context.compute_chunk_bounds(data.shape[0])
    ref_I, ref_D = ref_I[start:end], ref_D[start:end]
    assert sh_I.shape == (end - start, k)
    assert torch.allclose(sh_D, ref_D)
    differs = sh_I != ref_I
    assert torch.equal(sh_D[differs], ref_D[differs])  # every difference is a tie
    for row_sh, row_ref in zip(sh_I.tolist(), ref_I.tolist()):
        assert set(row_sh) == set(row_ref)


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

    def test_shard_matches_at_k_one(self, data, context):
        # k=1 is the smallest merge: a single nearest neighbor per query.
        _assert_matches_chunk(data, context, 1, "sqeuclidean")

    def test_shard_matches_full_neighborhood(self, data, context):
        # k just below the dataset size returns nearly every point, so each shard
        # contributes far more than k_search candidates and the merge spans the
        # tied tail. The neighbor set and distances must still be exact.
        _assert_neighbor_set_matches_chunk(data, context, N_SAMPLES - 1, "sqeuclidean")

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

    def test_small_batch_through_the_plan_matches(self, data, context):
        # The same bound reached end-to-end: an expert Flat config with a small
        # stream_batch_size drives the batching from the public plan API. As above,
        # the neighbor indices are exact and the distances match up to FAISS's
        # batch-tiling rounding.
        ref_D, ref_I = _reference(data, K, "sqeuclidean")
        plan = FaissPlanConfig(
            distribution="shard",
            expert=FaissConfig(index_type="Flat", stream_batch_size=11),
        )
        sh_D, sh_I = pairwise_distances(
            data,
            k=K,
            metric="sqeuclidean",
            backend=plan,
            return_indices=True,
            distributed_ctx=context,
        )
        start, end = context.compute_chunk_bounds(N_SAMPLES)
        assert torch.equal(sh_I, ref_I[start:end])
        assert torch.allclose(sh_D, ref_D[start:end], rtol=1e-4, atol=1e-4)
