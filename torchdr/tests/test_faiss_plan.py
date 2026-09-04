"""Tests for the high-level FAISS execution-plan API."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pickle
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchdr import EntropicAffinity, TSNE
from torchdr.distance import FaissConfig, FaissPlanConfig, pairwise_distances
from torchdr.distance.faiss_plan import (
    _AUTO_MEMORY_SAFETY,
    _estimate_search_peak_bytes,
    _resolve_faiss_plan,
    _select_auto_distribution,
)
from torchdr.utils import faiss


requires_faiss = pytest.mark.skipif(
    faiss is None or faiss is False, reason="faiss not installed"
)


class _FakeContext:
    """Minimal stand-in for DistributedContext during plan resolution.

    ``_resolve_faiss_plan`` only reads ``is_initialized`` and ``world_size``, so
    the topology decision can be exercised on a single process without a real
    process group.
    """

    def __init__(self, world_size, is_initialized=True):
        self.is_initialized = is_initialized
        self.world_size = world_size
        self.rank = 0
        self.local_rank = 0


def _budget_forcing(target, *, n_samples, n_features, world_size):
    """Per-GPU 'available' bytes that drives ``auto`` into ``target``.

    Derived from the same peak-memory model the selector uses, so the tests stay
    correct if the model constants change. The ``assert`` re-checks the crafted
    budget against the pure selector, failing at setup rather than misattributing
    a wrong branch.
    """
    replicate_peak = _estimate_search_peak_bytes(n_samples, n_features, n_samples)
    shard_rows = (n_samples + world_size - 1) // world_size
    shard_peak = _estimate_search_peak_bytes(n_samples, n_features, shard_rows)
    if target == "replicate":
        available = int(replicate_peak / _AUTO_MEMORY_SAFETY) + 10**9
    elif target == "shard":
        available = int((shard_peak + replicate_peak) / 2 / _AUTO_MEMORY_SAFETY)
    elif target == "refuse":
        available = int(shard_peak / _AUTO_MEMORY_SAFETY) - 10**6
    else:  # pragma: no cover - test wiring error
        raise ValueError(target)
    if target != "refuse":
        assert (
            _select_auto_distribution(
                n_samples=n_samples,
                n_features=n_features,
                world_size=world_size,
                available_memory_bytes=available,
            )
            == target
        )
    return available


@pytest.fixture(scope="module")
def data():
    return torch.randn(200, 8, generator=torch.Generator().manual_seed(0))


def test_config_defaults_are_exact_and_immutable():
    config = FaissPlanConfig()
    assert (config.mode, config.distribution, config.expert) == (
        "exact",
        "auto",
        None,
    )
    with pytest.raises(FrozenInstanceError):
        config.mode = "fast"
    assert pickle.loads(pickle.dumps(config)) == config


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"mode": "turbo"}, ValueError),
        ({"distribution": "single"}, ValueError),
        ({"expert": "not-a-config"}, TypeError),
        (
            {"mode": "fast", "expert": FaissConfig(index_type="IVF")},
            ValueError,
        ),
    ],
)
def test_config_rejects_invalid_combinations(kwargs, error):
    with pytest.raises(error):
        FaissPlanConfig(**kwargs)


@pytest.mark.parametrize(
    "config",
    [
        FaissPlanConfig(mode="balanced"),
        FaissPlanConfig(mode="fast"),
    ],
)
def test_unimplemented_intents_fail_explicitly(config):
    with pytest.raises(NotImplementedError):
        _resolve_faiss_plan(config)


def test_shard_and_replicate_resolve_across_a_group():
    ctx = _FakeContext(world_size=4)
    plan, resolved = _resolve_faiss_plan(
        FaissPlanConfig(distribution="shard"),
        n_samples=1_000,
        n_features=8,
        distributed_ctx=ctx,
    )
    assert plan.distribution == "shard"
    assert plan.index_type == resolved.index_type == "Flat"
    assert plan.index_memory_bytes == 250 * 8 * 4

    plan, _ = _resolve_faiss_plan(
        FaissPlanConfig(distribution="replicate"),
        n_samples=1_000,
        n_features=8,
        distributed_ctx=ctx,
    )
    assert plan.distribution == "replicate"


def test_shard_without_a_group_resolves_to_single():
    # Sharding needs more than one rank; a lone process just searches directly.
    plan, _ = _resolve_faiss_plan(FaissPlanConfig(distribution="shard"))
    assert plan.distribution == "single"

    plan, _ = _resolve_faiss_plan(
        FaissPlanConfig(distribution="shard"),
        distributed_ctx=_FakeContext(world_size=1),
    )
    assert plan.distribution == "single"


def test_auto_without_a_budget_preserves_replication():
    # The diagnostic ``faiss_plan_`` pass cannot measure memory, so it passes no
    # budget; auto must then keep the established replicated fast path rather than
    # guess. The distributed dispatcher supplies a measured budget separately.
    ctx = _FakeContext(world_size=2)
    plan, _ = _resolve_faiss_plan(
        FaissPlanConfig(distribution="auto"),
        n_samples=1_000,
        n_features=8,
        distributed_ctx=ctx,
    )
    assert plan.distribution == "replicate"


@pytest.mark.parametrize("world_size", [2, 4])
def test_select_auto_topology_follows_the_memory_budget(world_size):
    kw = dict(n_samples=100_000, n_features=128, world_size=world_size)
    assert _select_auto_distribution(available_memory_bytes=None, **kw) == "replicate"
    ample = _budget_forcing("replicate", **kw)
    assert _select_auto_distribution(available_memory_bytes=ample, **kw) == "replicate"
    tight = _budget_forcing("shard", **kw)
    assert _select_auto_distribution(available_memory_bytes=tight, **kw) == "shard"


def test_select_auto_refuses_when_neither_topology_fits():
    kw = dict(n_samples=100_000, n_features=128, world_size=2)
    starved = _budget_forcing("refuse", **kw)
    with pytest.raises(RuntimeError, match="run out of memory"):
        _select_auto_distribution(available_memory_bytes=starved, **kw)


def test_select_auto_keeps_replication_for_a_single_rank():
    # One rank cannot shard, so even a starving budget stays replicated; the
    # single-process path never launches the sharded collectives.
    assert (
        _select_auto_distribution(
            n_samples=100_000,
            n_features=128,
            world_size=1,
            available_memory_bytes=1,
        )
        == "replicate"
    )


def test_resolve_auto_shards_under_a_measured_budget():
    ctx = _FakeContext(world_size=4)
    budget = _budget_forcing("shard", n_samples=100_000, n_features=128, world_size=4)
    plan, resolved = _resolve_faiss_plan(
        FaissPlanConfig(distribution="auto"),
        n_samples=100_000,
        n_features=128,
        distributed_ctx=ctx,
        available_memory_bytes=budget,
    )
    assert plan.distribution == "shard"
    assert resolved.index_type == "Flat"
    # The reported index footprint follows the sharded, not the replicated, size.
    assert plan.index_memory_bytes == (100_000 // 4) * 128 * 4


def test_resolve_auto_refuses_an_over_budget_run():
    ctx = _FakeContext(world_size=2)
    budget = _budget_forcing("refuse", n_samples=100_000, n_features=128, world_size=2)
    with pytest.raises(RuntimeError, match="run out of memory"):
        _resolve_faiss_plan(
            FaissPlanConfig(distribution="auto"),
            n_samples=100_000,
            n_features=128,
            distributed_ctx=ctx,
            available_memory_bytes=budget,
        )


def test_resolve_auto_expert_index_stays_replicated_instead_of_refusing():
    # An expert (non-Flat) index cannot be sharded, so auto keeps it replicated
    # even under a budget that would otherwise shard or refuse an exact index.
    ctx = _FakeContext(world_size=2)
    plan, resolved = _resolve_faiss_plan(
        FaissPlanConfig(expert=FaissConfig(index_type="IVF")),
        n_samples=100_000,
        n_features=128,
        distributed_ctx=ctx,
        available_memory_bytes=1,
    )
    assert plan.distribution == "replicate"
    assert resolved.index_type == "IVF"


def test_explicit_shard_rejects_unsupported_expert_index():
    with pytest.raises(NotImplementedError, match="supports only exact Flat"):
        _resolve_faiss_plan(
            FaissPlanConfig(distribution="shard", expert=FaissConfig(index_type="IVF")),
            distributed_ctx=_FakeContext(world_size=2),
        )


def test_exact_plan_reports_resolved_execution():
    plan, resolved = _resolve_faiss_plan(
        FaissPlanConfig(), n_samples=1_000, n_features=8
    )
    assert plan.mode == "exact"
    assert plan.index_type == resolved.index_type == "Flat"
    assert plan.precision == "float32"
    assert plan.distribution == "single"
    assert plan.training_size == 0
    assert plan.index_memory_bytes == 1_000 * 8 * 4

    restored = pickle.loads(pickle.dumps(plan))
    assert restored == plan
    assert "index_memory_bytes=32000" in repr(restored)


def test_expert_resolution_is_non_mutating():
    expert = FaissConfig(
        index_type="IVFPQ", nlist=50, M=8, nbits=8, nprobe=4, custom_option=1
    )
    config = FaissPlanConfig(expert=expert)
    before = repr(expert)

    plan, resolved = _resolve_faiss_plan(config)

    assert repr(expert) == before
    assert resolved is not expert
    assert resolved.faiss_kwargs is not expert.faiss_kwargs
    assert resolved.faiss_kwargs == expert.faiss_kwargs
    assert plan.mode == "expert"
    assert plan.index_type == "IVFPQ"
    assert plan.precision == "reduced"
    assert plan.training_size is None


@requires_faiss
def test_exact_plan_matches_existing_faiss_backend(data):
    distances, indices = pairwise_distances(
        data, k=5, backend=FaissPlanConfig(), return_indices=True
    )
    reference_distances, reference_indices = pairwise_distances(
        data, k=5, backend="faiss", return_indices=True
    )
    assert torch.equal(indices, reference_indices)
    assert torch.allclose(distances, reference_distances)


@requires_faiss
def test_plan_accepts_dataloader_input(data):
    loader = DataLoader(TensorDataset(data), batch_size=32, shuffle=False)
    distances, indices = pairwise_distances(
        loader, k=5, backend=FaissPlanConfig(), return_indices=True
    )
    assert distances.shape == indices.shape == (len(data), 5)


def test_shard_plan_rejects_dataloader_instead_of_silently_replicating(data):
    loader = DataLoader(TensorDataset(data), batch_size=32, shuffle=False)
    with pytest.raises(NotImplementedError, match="does not yet support DataLoader"):
        pairwise_distances(
            loader,
            k=5,
            backend=FaissPlanConfig(distribution="shard"),
            distributed_ctx=_FakeContext(world_size=2),
        )


@requires_faiss
def test_plan_is_exposed_by_affinity_and_estimator(data):
    affinity = EntropicAffinity(perplexity=15, backend=FaissPlanConfig(), sparsity=True)
    affinity(data)
    assert affinity.faiss_plan_.index_type == "Flat"

    estimator = TSNE(
        perplexity=15,
        backend=FaissPlanConfig(),
        max_iter=0,
        random_state=0,
    ).fit(data)
    assert estimator.faiss_plan_ == estimator.affinity_in.faiss_plan_
