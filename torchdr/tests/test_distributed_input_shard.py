"""Real-process tests for exact k-NN over an input whose rows are sharded.

Unlike ``test_distributed_faiss_shard`` -- where every rank holds the full input
and only the FAISS index is sharded -- here each rank holds a *distinct* shard of
the rows and concatenating the shards in rank order reconstructs the dataset.
This is the layout that lowers the per-rank input footprint.

Ordinary CI runs these against Gloo and CPU FAISS. The same module can run with
NCCL and GPU FAISS, because TorchDR selects the process-group backend from the
available device. Each rank's result must equal single-process Flat search over
the reconstructed global dataset, restricted to that rank's rows.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr.distance import FaissConfig, pairwise_distances
from torchdr.distance.faiss import input_sharded_pairwise_distances_faiss
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

    Fails loudly on a single process: with one rank the whole dataset is the only
    shard and the merge never crosses a rank boundary, so a silent one-rank run
    would report green while exercising none of the sharded search.
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


def _global_dataset(n_samples, seed=0):
    """The reconstructed global input, identical on every rank via the seed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, N_FEATURES, generator=generator)


def _even_counts(n_samples, world_size):
    """Balanced split: near-equal rows per rank."""
    base, extra = divmod(n_samples, world_size)
    return [base + (1 if r < extra else 0) for r in range(world_size)]


def _weighted_counts(n_samples, world_size):
    """Deliberately uneven split with strictly increasing, distinct rank sizes.

    The offsets this produces do not line up with the balanced partitioner, so a
    match against the oracle proves the global ids come from the declared layout
    rather than from ``chunk_bounds`` re-slicing a replicated copy.
    """
    weights = [r + 1 for r in range(world_size)]
    total = sum(weights)
    counts = [n_samples * w // total for w in weights]
    counts[-1] += n_samples - sum(counts)
    return counts


def _empty_first_counts(n_samples, world_size):
    """Uneven split where rank 0 owns zero rows."""
    counts = _weighted_counts(n_samples, world_size)
    counts[-1] += counts[0]
    counts[0] = 0
    return counts


def _local_shard(full, counts, rank):
    """This rank's contiguous shard plus its global [start, end) range."""
    start = int(sum(counts[:rank]))
    end = start + int(counts[rank])
    return full[start:end].contiguous(), start, end


def _reference(data, k, metric, exclude_diag=False):
    """Single-process exact Flat neighbors over the reconstructed dataset."""
    return pairwise_distances(
        data,
        k=k,
        metric=metric,
        backend=FaissConfig(),
        exclude_diag=exclude_diag,
        return_indices=True,
    )


def _assert_matches_global(full, counts, context, k, metric, exclude_diag=False):
    X_local, start, end = _local_shard(full, counts, context.rank)
    ref_D, ref_I = _reference(full, k, metric, exclude_diag)
    sh_D, sh_I = input_sharded_pairwise_distances_faiss(
        X_local,
        k=k,
        metric=metric,
        exclude_diag=exclude_diag,
        distributed_ctx=context,
    )
    assert sh_I.shape == (end - start, k)
    assert torch.equal(sh_I, ref_I[start:end])
    assert torch.allclose(sh_D, ref_D[start:end], rtol=1e-4, atol=1e-4)


class TestInputShardedSearch:
    @pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean", "angular"])
    def test_even_shard_matches_single_process_flat(self, context, metric):
        # The heart of step 1 of #301: with the rows split across ranks and no
        # rank holding the full input, the merged neighbors must reproduce the
        # single-process global result for every metric.
        full = _global_dataset(N_SAMPLES)
        counts = _even_counts(N_SAMPLES, context.world_size)
        _assert_matches_global(full, counts, context, K, metric)

    @pytest.mark.parametrize("metric", ["sqeuclidean", "angular"])
    def test_uneven_shard_matches_single_process_flat(self, context, metric):
        # Distinct, non-balanced shard sizes give offsets that do not match the
        # balanced partitioner, so the global ids must come from the declared
        # layout for the merge to line up with the oracle.
        full = _global_dataset(N_SAMPLES, seed=1)
        counts = _weighted_counts(N_SAMPLES, context.world_size)
        _assert_matches_global(full, counts, context, K, metric)

    def test_shard_excludes_self_neighbor(self, context):
        # exclude_diag removes each query's own global row, matched by the global
        # id from the layout even though a query sits in its own rank's shard.
        full = _global_dataset(N_SAMPLES, seed=2)
        counts = _weighted_counts(N_SAMPLES, context.world_size)
        _assert_matches_global(full, counts, context, K, "sqeuclidean", True)

    def test_empty_rank_participates(self, context):
        # A rank that owns zero rows must still take part in every collective:
        # it indexes nothing, contributes no candidates, and returns an empty
        # (0, k) result, while the other ranks still get the exact neighbors.
        full = _global_dataset(N_SAMPLES, seed=3)
        counts = _empty_first_counts(N_SAMPLES, context.world_size)
        _assert_matches_global(full, counts, context, K, "sqeuclidean")
        _assert_matches_global(full, counts, context, K, "angular")

    def test_k_larger_than_a_shard(self, context):
        # With few rows per shard each local search returns padding (-1) that the
        # merge must discard while still assembling the correct k global
        # neighbors from across the shards.
        full = _global_dataset(3 * context.world_size + 1, seed=4)
        counts = _weighted_counts(full.shape[0], context.world_size)
        _assert_matches_global(full, counts, context, K, "sqeuclidean")
        _assert_matches_global(full, counts, context, K, "sqeuclidean", True)

    def test_bounded_query_batch_matches(self, context):
        # A tiny query batch forces many broadcast/all_gather rounds; the neighbor
        # identities must be identical, which is what keeps the merge memory
        # bounded by the batch rather than the shard. FAISS's blocked matmul makes
        # the raw distances match only up to a ~1e-6 wobble, so the ids are exact
        # while the distances are compared with a tolerance.
        full = _global_dataset(N_SAMPLES, seed=5)
        counts = _weighted_counts(N_SAMPLES, context.world_size)
        X_local, start, end = _local_shard(full, counts, context.rank)
        ref_D, ref_I = _reference(full, K, "sqeuclidean")
        sh_D, sh_I = input_sharded_pairwise_distances_faiss(
            X_local,
            k=K,
            metric="sqeuclidean",
            distributed_ctx=context,
            query_batch_size=7,
        )
        assert torch.equal(sh_I, ref_I[start:end])
        assert torch.allclose(sh_D, ref_D[start:end], rtol=1e-4, atol=1e-4)


class TestShardLayout:
    """The metadata contract that the sharded search is built on."""

    def test_layout_reports_rank_major_offsets(self, context):
        counts = _weighted_counts(N_SAMPLES, context.world_size)
        rank = context.rank
        start = int(sum(counts[:rank]))
        X_local = torch.randn(counts[rank], N_FEATURES)
        layout = gather_shard_layout(X_local, context)
        assert layout.rank == rank
        assert layout.world_size == context.world_size
        assert layout.local_count == counts[rank]
        assert layout.global_count == N_SAMPLES
        assert layout.local_offset == start
        assert list(layout.counts) == counts
        assert torch.equal(
            layout.query_ids(), torch.arange(start, start + counts[rank])
        )

    def test_layout_supports_empty_rank(self, context):
        counts = _empty_first_counts(N_SAMPLES, context.world_size)
        rank = context.rank
        X_local = torch.randn(counts[rank], N_FEATURES)
        layout = gather_shard_layout(X_local, context)
        assert layout.local_count == counts[rank]
        assert layout.global_count == N_SAMPLES
        assert list(layout.counts) == counts
        assert layout.query_ids().numel() == counts[rank]

    def test_layout_rejects_feature_mismatch(self, context):
        # One rank declares a different feature count. The check runs on the full
        # gathered vector, so every rank raises together rather than silently
        # returning wrong neighbors.
        n_features = N_FEATURES + (1 if context.rank == context.world_size - 1 else 0)
        X_local = torch.randn(4, n_features)
        with pytest.raises(ValueError, match="disagree on n_features"):
            gather_shard_layout(X_local, context)


class TestInputShardedFailurePropagation:
    """An index-build or search failure stops every rank symmetrically."""

    def test_index_build_failure_raises_everywhere(self, context, monkeypatch):
        import torchdr.distance.faiss as faiss_mod

        failing_rank = context.world_size - 1
        real_create_index = faiss_mod._create_index

        def faulty_create_index(*args, **kwargs):
            if context.rank == failing_rank:
                raise RuntimeError("injected local FAISS index build failure")
            return real_create_index(*args, **kwargs)

        monkeypatch.setattr(faiss_mod, "_create_index", faulty_create_index)

        full = _global_dataset(N_SAMPLES)
        counts = _even_counts(N_SAMPLES, context.world_size)
        X_local, _, _ = _local_shard(full, counts, context.rank)
        with pytest.raises(
            RuntimeError, match="a rank failed to build its local FAISS index"
        ):
            input_sharded_pairwise_distances_faiss(
                X_local, k=K, metric="sqeuclidean", distributed_ctx=context
            )
        dist.barrier()

    def test_local_search_failure_raises_everywhere(self, context, monkeypatch):
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

        full = _global_dataset(N_SAMPLES)
        counts = _even_counts(N_SAMPLES, context.world_size)
        X_local, _, _ = _local_shard(full, counts, context.rank)
        with pytest.raises(RuntimeError, match="a rank failed its local FAISS search"):
            input_sharded_pairwise_distances_faiss(
                X_local, k=K, metric="sqeuclidean", distributed_ctx=context
            )
        dist.barrier()
