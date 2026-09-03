"""Tests for the high-level FAISS execution-plan API (FaissPlanConfig)."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pickle
from dataclasses import FrozenInstanceError

import pytest
import torch

from torchdr import EntropicAffinity
from torchdr.distance import FaissConfig, FaissPlanConfig, pairwise_distances
from torchdr.distance.faiss_plan import _FaissPlan, resolve_faiss_plan
from torchdr.utils import faiss


pytestmark = pytest.mark.skipif(
    faiss is None or faiss is False, reason="faiss not installed"
)

N_SAMPLES = 200
N_FEATURES = 8


@pytest.fixture(scope="module")
def data():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N_SAMPLES, N_FEATURES, generator=generator)


class TestFaissPlanConfigValidation:
    """Construction-time validation of the public intent config."""

    def test_defaults(self):
        cfg = FaissPlanConfig()
        assert cfg.mode == "exact"
        assert cfg.distribution == "auto"
        assert cfg.memory_budget == "auto"
        assert cfg.random_state is None
        assert cfg.expert is None

    def test_repr_is_readable(self):
        text = repr(FaissPlanConfig(random_state=3, expert=FaissConfig()))
        assert text.startswith("FaissPlanConfig(")
        assert "random_state=3" in text

    @pytest.mark.parametrize("mode", ["turbo", "Exact", "", None])
    def test_invalid_mode_raises(self, mode):
        with pytest.raises(ValueError, match="mode"):
            FaissPlanConfig(mode=mode)

    @pytest.mark.parametrize("distribution", ["sharded", "single", None])
    def test_invalid_distribution_raises(self, distribution):
        with pytest.raises(ValueError, match="distribution"):
            FaissPlanConfig(distribution=distribution)

    @pytest.mark.parametrize("budget", ["huge", -1, 0, 1.5, True])
    def test_invalid_memory_budget_raises(self, budget):
        with pytest.raises(ValueError, match="memory_budget"):
            FaissPlanConfig(memory_budget=budget)

    def test_valid_explicit_memory_budget_constructs(self):
        # A positive integer budget is a valid *field* (resolution raises later).
        cfg = FaissPlanConfig(memory_budget=2**30)
        assert cfg.memory_budget == 2**30

    def test_expert_must_be_faissconfig(self):
        with pytest.raises(TypeError, match="expert"):
            FaissPlanConfig(expert="not-a-config")

    def test_expert_with_nondefault_mode_raises(self):
        with pytest.raises(ValueError, match="expert override"):
            FaissPlanConfig(mode="fast", expert=FaissConfig())


class TestResolveFaissPlan:
    """Resolution of intent into an immutable plan + low-level FaissConfig."""

    def test_exact_resolves_to_flat_full_precision(self):
        plan, resolved = resolve_faiss_plan(FaissPlanConfig(mode="exact"))
        assert isinstance(plan, _FaissPlan)
        assert plan.index_type == "Flat"
        assert plan.precision == "float32"
        assert plan.distribution == "replicate"
        assert plan.training_size == 0
        assert isinstance(resolved, FaissConfig)
        assert resolved.index_type == "Flat"

    def test_auto_distribution_resolves_to_replicate(self):
        plan, _ = resolve_faiss_plan(FaissPlanConfig(distribution="auto"))
        assert plan.distribution == "replicate"
        plan2, _ = resolve_faiss_plan(FaissPlanConfig(distribution="replicate"))
        assert plan2.distribution == "replicate"

    def test_memory_estimate_when_shape_known(self):
        plan, _ = resolve_faiss_plan(FaissPlanConfig(), n_samples=1000, dim=N_FEATURES)
        assert plan.memory_estimate == 1000 * N_FEATURES * 4

    def test_memory_estimate_unknown_without_shape(self):
        plan, _ = resolve_faiss_plan(FaissPlanConfig())
        assert plan.memory_estimate is None

    def test_random_state_recorded(self):
        plan, _ = resolve_faiss_plan(FaissPlanConfig(random_state=7))
        assert plan.random_state == 7

    def test_expert_override_used_verbatim(self):
        expert = FaissConfig(index_type="IVFPQ", nlist=50, M=8, nbits=8, nprobe=4)
        plan, resolved = resolve_faiss_plan(FaissPlanConfig(expert=expert))
        assert resolved.index_type == "IVFPQ"
        assert resolved.nlist == 50
        assert resolved.M == 8
        assert resolved.nprobe == 4
        assert plan.index_type == "IVFPQ"
        assert plan.precision == "reduced"  # PQ compresses
        assert plan.training_size is None  # resolved downstream

    def test_expert_ivf_is_full_precision(self):
        plan, _ = resolve_faiss_plan(
            FaissPlanConfig(expert=FaissConfig(index_type="IVF"))
        )
        assert plan.precision == "float32"

    @pytest.mark.parametrize("mode", ["balanced", "fast"])
    def test_presets_not_yet_supported(self, mode):
        with pytest.raises(NotImplementedError, match="#304"):
            resolve_faiss_plan(FaissPlanConfig(mode=mode))

    def test_shard_not_yet_supported(self):
        with pytest.raises(NotImplementedError, match="#301"):
            resolve_faiss_plan(FaissPlanConfig(distribution="shard"))

    def test_explicit_memory_budget_not_yet_supported(self):
        with pytest.raises(NotImplementedError, match="#301"):
            resolve_faiss_plan(FaissPlanConfig(memory_budget=2**30))

    def test_exact_never_approximates(self):
        # No combination of the supported knobs may resolve to an approximate
        # index unless the user explicitly supplied an approximate expert config.
        for cfg in (
            FaissPlanConfig(),
            FaissPlanConfig(distribution="replicate"),
            FaissPlanConfig(random_state=1),
        ):
            _, resolved = resolve_faiss_plan(cfg)
            assert resolved.index_type == "Flat"

    def test_non_mutation_of_user_config(self):
        expert = FaissConfig(index_type="IVFPQ", nlist=42, M=8)
        cfg = FaissPlanConfig(random_state=1, expert=expert)
        before_cfg, before_expert = repr(cfg), repr(expert)

        _, resolved = resolve_faiss_plan(cfg, n_samples=100, dim=N_FEATURES)

        assert repr(cfg) == before_cfg
        assert repr(expert) == before_expert
        # The resolved config is a fresh object, not the user's expert instance.
        assert resolved is not expert


class TestFaissPlanImmutabilityAndSerialization:
    """The resolved plan is frozen, reproducibly represented, and picklable."""

    def test_plan_is_frozen(self):
        plan, _ = resolve_faiss_plan(FaissPlanConfig())
        with pytest.raises(FrozenInstanceError):
            plan.index_type = "IVF"

    def test_plan_repr(self):
        plan, _ = resolve_faiss_plan(FaissPlanConfig(), n_samples=10, dim=4)
        text = repr(plan)
        assert text.startswith("FaissPlan(")
        assert "index_type='Flat'" in text
        assert "memory_estimate=160 bytes" in text

    def test_plan_repr_unknown_fields(self):
        plan, _ = resolve_faiss_plan(FaissPlanConfig())
        text = repr(plan)
        assert "memory_estimate=unknown" in text

    def test_plan_is_picklable(self):
        plan, _ = resolve_faiss_plan(FaissPlanConfig(), n_samples=10, dim=4)
        restored = pickle.loads(pickle.dumps(plan))
        assert restored == plan
        assert repr(restored) == repr(plan)


class TestFaissPlanIntegration:
    """End-to-end behavior through pairwise_distances and an affinity (CPU)."""

    def test_pairwise_distances_accepts_plan_config(self, data):
        # Direct-call guard: a plan config resolves to the exact Flat backend and
        # returns the same neighbors as backend="faiss" (CPU-only fallback).
        d_ref, i_ref = pairwise_distances(
            data, k=5, backend="faiss", return_indices=True
        )
        d_plan, i_plan = pairwise_distances(
            data, k=5, backend=FaissPlanConfig(mode="exact"), return_indices=True
        )
        assert torch.equal(i_ref, i_plan)
        assert torch.allclose(d_ref, d_plan)

    def test_pairwise_distances_reproducible(self, data):
        _, i_a = pairwise_distances(
            data, k=5, backend=FaissPlanConfig(mode="exact"), return_indices=True
        )
        _, i_b = pairwise_distances(
            data, k=5, backend=FaissPlanConfig(mode="exact"), return_indices=True
        )
        assert torch.equal(i_a, i_b)

    def test_affinity_records_plan_and_matches_faiss(self, data):
        aff_plan = EntropicAffinity(
            perplexity=15,
            backend=FaissPlanConfig(mode="exact"),
            sparsity=True,
            verbose=False,
        )
        _, idx_plan = aff_plan(data)

        # The resolved plan is exposed for inspection after computation.
        assert hasattr(aff_plan, "faiss_plan_")
        assert aff_plan.faiss_plan_.index_type == "Flat"
        assert aff_plan.faiss_plan_.precision == "float32"

        aff_faiss = EntropicAffinity(
            perplexity=15,
            backend="faiss",
            sparsity=True,
            verbose=False,
        )
        _, idx_faiss = aff_faiss(data)

        # Exact plan == plain FAISS Flat: identical k-NN structure.
        assert torch.equal(idx_plan, idx_faiss)

    def test_affinity_without_plan_has_no_attr(self, data):
        aff = EntropicAffinity(
            perplexity=15, backend="faiss", sparsity=True, verbose=False
        )
        aff(data)
        assert not hasattr(aff, "faiss_plan_")
