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

import pytest
import torch
import torch.distributed as dist

from torchdr.distance import FaissConfig, FaissPlanConfig, pairwise_distances
from torchdr.distance import base as distance_base
from torchdr.distance.faiss import sharded_pairwise_distances_faiss
from torchdr.distance.faiss_plan import (
    _AUTO_MEMORY_SAFETY,
    _estimate_search_peak_bytes,
)
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


def _budget_forcing(target, world_size, n_samples=N_SAMPLES, n_features=N_FEATURES):
    """Per-GPU 'available' bytes that drives ``auto`` into ``target``.

    Computed from the peak-memory model the selector uses so the tests track the
    model rather than a hard-coded byte count.
    """
    replicate_peak = _estimate_search_peak_bytes(n_samples, n_features, n_samples)
    shard_rows = (n_samples + world_size - 1) // world_size
    shard_peak = _estimate_search_peak_bytes(n_samples, n_features, shard_rows)
    if target == "replicate":
        return int(replicate_peak / _AUTO_MEMORY_SAFETY) + 10**9
    if target == "shard":
        return int((shard_peak + replicate_peak) / 2 / _AUTO_MEMORY_SAFETY)
    if target == "refuse":
        return int(shard_peak / _AUTO_MEMORY_SAFETY) - 10**6
    raise ValueError(target)


def _auto(data, context, exclude_diag=False):
    return pairwise_distances(
        data,
        k=K,
        metric="sqeuclidean",
        backend=FaissPlanConfig(distribution="auto"),
        exclude_diag=exclude_diag,
        return_indices=True,
        distributed_ctx=context,
    )


def _spy_on_shard(monkeypatch):
    """Count how often this rank routes through the sharded search."""
    calls = {"n": 0}
    real = distance_base.sharded_pairwise_distances_faiss

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(distance_base, "sharded_pairwise_distances_faiss", counting)
    return calls


def _all_gather_int(value, world_size):
    local = torch.tensor([int(value)], dtype=torch.int64)
    gathered = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [int(t.item()) for t in gathered]


class TestAutoDistribution:
    """End-to-end memory-aware ``distribution='auto'`` selection (issue #301).

    The per-GPU memory read is replaced by an injected budget so the selection
    logic -- which normally only fires on a real multi-GPU node -- is exercised
    on CPU/Gloo CI. Every case still runs the real process group and the real
    exact-search paths, and compares against single-process Flat neighbors.
    """

    def test_auto_replicates_when_memory_is_ample(self, data, context, monkeypatch):
        # A budget that fits a full index keeps the fast replicated path: no rank
        # enters the sharded search, and every chunk still matches the reference.
        monkeypatch.setattr(
            distance_base,
            "_available_gpu_memory_bytes",
            lambda ctx: _budget_forcing("replicate", context.world_size),
        )
        calls = _spy_on_shard(monkeypatch)
        ref_D, ref_I = _reference(data, K, "sqeuclidean")
        au_D, au_I = _auto(data, context)
        start, end = context.compute_chunk_bounds(N_SAMPLES)
        assert torch.equal(au_I, ref_I[start:end])
        assert torch.allclose(au_D, ref_D[start:end])
        assert (
            _all_gather_int(calls["n"], context.world_size) == [0] * context.world_size
        )

    def test_auto_shards_when_replication_exceeds_memory(
        self, data, context, monkeypatch
    ):
        # A budget too small for a full index but large enough sharded routes every
        # rank through the sharded search and still reproduces the exact neighbors.
        monkeypatch.setattr(
            distance_base,
            "_available_gpu_memory_bytes",
            lambda ctx: _budget_forcing("shard", context.world_size),
        )
        calls = _spy_on_shard(monkeypatch)
        ref_D, ref_I = _reference(data, K, "sqeuclidean")
        au_D, au_I = _auto(data, context)
        start, end = context.compute_chunk_bounds(N_SAMPLES)
        assert torch.equal(au_I, ref_I[start:end])
        assert torch.allclose(au_D, ref_D[start:end])
        assert (
            _all_gather_int(calls["n"], context.world_size) == [1] * context.world_size
        )

    def test_auto_decision_is_consistent_across_divergent_budgets(
        self, data, context, monkeypatch
    ):
        # Rank 0 alone would replicate (ample budget); the others would shard. The
        # cross-rank MIN reduce forces one shared decision -- had the ranks decided
        # locally, some would launch the sharded collectives while others would not
        # and the run would deadlock. Reaching the assertions at all proves the
        # decision was uniform; the spy confirms every rank took the sharded path.
        def divergent(ctx):
            target = "replicate" if ctx.rank == 0 else "shard"
            return _budget_forcing(target, context.world_size)

        monkeypatch.setattr(distance_base, "_available_gpu_memory_bytes", divergent)
        calls = _spy_on_shard(monkeypatch)
        ref_D, ref_I = _reference(data, K, "sqeuclidean")
        au_D, au_I = _auto(data, context)
        start, end = context.compute_chunk_bounds(N_SAMPLES)
        assert torch.equal(au_I, ref_I[start:end])
        assert torch.allclose(au_D, ref_D[start:end])
        assert (
            _all_gather_int(calls["n"], context.world_size) == [1] * context.world_size
        )

    def test_auto_refuses_consistently_when_nothing_fits(
        self, data, context, monkeypatch
    ):
        # When even a sharded index is over budget, every rank must raise -- not
        # hang -- so the failure surfaces in lockstep after the shared reduce.
        monkeypatch.setattr(
            distance_base,
            "_available_gpu_memory_bytes",
            lambda ctx: _budget_forcing("refuse", context.world_size),
        )
        with pytest.raises(RuntimeError, match="run out of memory"):
            _auto(data, context)
        dist.barrier()
