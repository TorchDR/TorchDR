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
from torchdr.distance.faiss_plan import _choose_distribution, _resolve_faiss_plan
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


def test_auto_shards_only_when_the_index_will_not_fit():
    ctx = _FakeContext(world_size=2)
    common = dict(n_samples=1_000, n_features=8, distributed_ctx=ctx)

    fits, _ = _resolve_faiss_plan(
        FaissPlanConfig(distribution="auto"),
        available_memory_bytes=10**9,
        **common,
    )
    assert fits.distribution == "replicate"

    too_big, _ = _resolve_faiss_plan(
        FaissPlanConfig(distribution="auto"),
        available_memory_bytes=1_000,
        **common,
    )
    assert too_big.distribution == "shard"

    # Without a per-rank budget the safe default is replication.
    unknown, _ = _resolve_faiss_plan(FaissPlanConfig(distribution="auto"), **common)
    assert unknown.distribution == "replicate"


def test_choose_distribution_never_replicates_an_index_that_will_not_fit():
    assert (
        _choose_distribution(
            index_memory_bytes=100, available_memory_bytes=1_000, world_size=2
        )
        == "replicate"
    )
    assert (
        _choose_distribution(
            index_memory_bytes=2_000, available_memory_bytes=1_000, world_size=2
        )
        == "shard"
    )
    # A missing budget or a single rank falls back to replication rather than
    # silently sharding, and never claims a giant index fits.
    assert (
        _choose_distribution(
            index_memory_bytes=10**12, available_memory_bytes=None, world_size=8
        )
        == "replicate"
    )
    assert (
        _choose_distribution(
            index_memory_bytes=10**12, available_memory_bytes=1, world_size=1
        )
        == "replicate"
    )
    # The safety margin tips a just-barely-fitting index into sharding.
    assert (
        _choose_distribution(
            index_memory_bytes=1_000,
            available_memory_bytes=1_000,
            world_size=2,
            safety_fraction=0.2,
        )
        == "shard"
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
