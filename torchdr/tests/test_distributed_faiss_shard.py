"""Real-process tests for exact k-NN over a database sharded across ranks.

Ordinary CI runs these against Gloo and CPU FAISS. The collective implementation
also supports NCCL and GPU FAISS when given CUDA-resident inputs; that path is
validated separately on GPU hardware. Each rank's result must equal replicated
Flat search for its query chunk.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr.distance import FaissConfig, FaissPlanConfig, pairwise_distances
from torchdr.distance.faiss import (
    input_sharded_pairwise_distances_faiss,
    sharded_pairwise_distances_faiss,
)
from torchdr.distributed import (
    DistributedContext,
    init_distributed,
    shutdown_distributed,
)
from torchdr.distributed.input_contract import gather_shard_layout
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


def _weighted_counts(n_samples, world_size):
    """Uneven rank-major partition that differs from the balanced partitioner."""
    weights = [rank + 1 for rank in range(world_size)]
    total = sum(weights)
    counts = [n_samples * weight // total for weight in weights]
    counts[-1] += n_samples - sum(counts)
    return counts


def _local_shard(data, counts, rank):
    start = sum(counts[:rank])
    end = start + counts[rank]
    return data[start:end].contiguous(), start, end


def _assert_input_shard_matches(
    data, counts, context, k, metric, exclude_diag=False, query_batch_size=None
):
    """Compare a distinct local input shard with the global Flat oracle."""
    X_local, start, end = _local_shard(data, counts, context.rank)
    ref_D, ref_I = _reference(data, k, metric, exclude_diag)
    sh_D, sh_I = input_sharded_pairwise_distances_faiss(
        X_local,
        k=k,
        metric=metric,
        exclude_diag=exclude_diag,
        distributed_ctx=context,
        query_batch_size=query_batch_size,
    )
    assert sh_I.shape == (end - start, k)
    assert torch.equal(sh_I, ref_I[start:end])
    assert torch.allclose(sh_D, ref_D[start:end], rtol=1e-4, atol=1e-4)


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


class TestInputShardedSearch:
    """Exact search when every rank holds only its distinct local rows."""

    @pytest.mark.parametrize("metric", ["sqeuclidean", "angular"])
    def test_uneven_input_shard_matches_global_flat(self, context, metric):
        # Uneven offsets prove that global ids come from the declared layout,
        # rather than the balanced partition used for replicated inputs.
        data = _dataset(N_SAMPLES, seed=11)
        counts = _weighted_counts(N_SAMPLES, context.world_size)
        _assert_input_shard_matches(data, counts, context, K, metric)

    def test_global_self_neighbor_is_excluded(self, context):
        data = _dataset(N_SAMPLES, seed=12)
        counts = _weighted_counts(N_SAMPLES, context.world_size)
        _assert_input_shard_matches(
            data, counts, context, K, "sqeuclidean", exclude_diag=True
        )

    def test_empty_rank_participates(self, context):
        data = _dataset(N_SAMPLES, seed=13)
        counts = _weighted_counts(N_SAMPLES, context.world_size)
        counts[-1] += counts[0]
        counts[0] = 0
        _assert_input_shard_matches(data, counts, context, K, "angular")

    def test_k_larger_than_local_shard(self, context):
        data = _dataset(3 * context.world_size + 1, seed=14)
        counts = _weighted_counts(data.shape[0], context.world_size)
        _assert_input_shard_matches(data, counts, context, K, "sqeuclidean")

    def test_bounded_query_batch_matches(self, context):
        data = _dataset(N_SAMPLES, seed=15)
        counts = _weighted_counts(N_SAMPLES, context.world_size)
        X_local, start, end = _local_shard(data, counts, context.rank)
        ref_D, _ = _reference(data, K, "sqeuclidean")
        sh_D, sh_I = input_sharded_pairwise_distances_faiss(
            X_local,
            k=K,
            metric="sqeuclidean",
            distributed_ctx=context,
            query_batch_size=7,
        )
        # FAISS's blocked float32 kernel can swap numerically tied neighbors when
        # the query batch changes, so the meaningful invariant here is the exact
        # top-k distance profile. Other tests above require exact global ids.
        assert sh_I.shape == (end - start, K)
        assert ((sh_I >= 0) & (sh_I < N_SAMPLES)).all()
        assert torch.allclose(sh_D, ref_D[start:end], rtol=1e-4, atol=1e-4)

    def test_layout_rejects_feature_mismatch_collectively(self, context):
        n_features = N_FEATURES + (context.rank == context.world_size - 1)
        X_local = torch.randn(4, n_features)
        with pytest.raises(ValueError, match="disagree on n_features"):
            gather_shard_layout(X_local, context)


class TestShardedSearchFailurePropagation:
    """A local search failure stops every rank before result gathering."""

    def test_local_search_failure_raises_everywhere(self, data, context, monkeypatch):
        import torchdr.distance.faiss as faiss_mod

        failing_rank = context.world_size - 1
        real_create_index = faiss_mod._create_index

        class _FailingSearchIndex:
            def __init__(self, inner):
                self._inner = inner

            def add(self, x):
                return self._inner.add(x)

            def search(self, *args, **kwargs):
                raise RuntimeError("injected local FAISS search failure")

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def faulty_create_index(*args, **kwargs):
            index = real_create_index(*args, **kwargs)
            if context.rank == failing_rank:
                return _FailingSearchIndex(index)
            return index

        monkeypatch.setattr(faiss_mod, "_create_index", faulty_create_index)

        with pytest.raises(RuntimeError, match="a rank failed its local FAISS search"):
            sharded_pairwise_distances_faiss(
                data, k=K, metric="sqeuclidean", distributed_ctx=context
            )

        # The process group remains usable after the symmetric failure.
        dist.barrier()


class TestInputShardedFailurePropagation:
    """A local index-build failure stops every rank before search begins."""

    def test_index_build_failure_raises_everywhere(self, data, context, monkeypatch):
        import torchdr.distance.faiss as faiss_mod

        failing_rank = context.world_size - 1
        real_create_index = faiss_mod._create_index

        def faulty_create_index(*args, **kwargs):
            if context.rank == failing_rank:
                raise RuntimeError("injected local FAISS index build failure")
            return real_create_index(*args, **kwargs)

        monkeypatch.setattr(faiss_mod, "_create_index", faulty_create_index)
        counts = _weighted_counts(data.shape[0], context.world_size)
        X_local, _, _ = _local_shard(data, counts, context.rank)

        with pytest.raises(
            RuntimeError, match="a rank failed to build its local FAISS index"
        ):
            input_sharded_pairwise_distances_faiss(
                X_local, k=K, metric="sqeuclidean", distributed_ctx=context
            )

        dist.barrier()
