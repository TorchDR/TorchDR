"""Tests for the distributed input contract."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import torchdr.distributed.input_contract as contract
from torchdr.distributed import DistributedContext, validate_distributed_input
from torchdr.distributed.input_contract import _local_metadata, _tensor_checksum


def _context(rank, world_size):
    """DistributedContext describing a world of ``world_size`` ranks."""
    ctx = DistributedContext(force_enable=True)
    ctx.rank = rank
    ctx.world_size = world_size
    ctx.local_rank = 0
    return ctx


@pytest.fixture
def simulate_world(monkeypatch):
    """Run one rank of a simulated world inside a single process.

    ``validate_distributed_input`` learns about the other ranks only through
    ``all_gather_into_tensor``. Replacing that collective with the metadata of
    a scripted list of per-rank inputs exercises the full comparison logic
    deterministically, on CPU, with no process group.
    """

    def _simulate(inputs, rank=0, verify_content=True):
        ctx = _context(rank, len(inputs))
        # Patched after building the context: DistributedContext reads these
        # too, and would otherwise look for a real process group.
        monkeypatch.setattr(contract.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(contract.dist, "get_backend", lambda: "gloo")
        monkeypatch.setattr(contract.dist, "get_rank", lambda *a, **k: rank)
        monkeypatch.setattr(
            contract.dist, "get_world_size", lambda *a, **k: len(inputs)
        )

        def fake_all_gather_into_tensor(output, input_, *args, **kwargs):
            view = output.view(len(inputs), -1)
            for r, peer in enumerate(inputs):
                if r == rank:
                    view[r] = input_
                else:
                    view[r] = torch.tensor(
                        _local_metadata(
                            peer,
                            contract._loader_shard_info(peer) is not None,
                            verify_content,
                        ),
                        dtype=torch.int64,
                    )

        monkeypatch.setattr(
            contract.dist, "all_gather_into_tensor", fake_all_gather_into_tensor
        )
        return validate_distributed_input(
            inputs[rank], ctx, verify_content=verify_content
        )

    return _simulate


def _loader(X, batch_size=8, **kwargs):
    return DataLoader(TensorDataset(X), batch_size=batch_size, **kwargs)


class TestChecksum:
    """Unit tests for the content checksum."""

    def test_deterministic(self):
        """Repeated calls on the same tensor agree."""
        X = torch.randn(64, 5)
        assert _tensor_checksum(X) == _tensor_checksum(X)

    def test_detects_permutation(self):
        """Row order changes the checksum, so shuffling is caught."""
        X = torch.randn(64, 5)
        perm = torch.randperm(64, generator=torch.Generator().manual_seed(0))
        assert _tensor_checksum(X[perm]) != _tensor_checksum(X)

    def test_detects_single_value_change(self):
        """A one-element edit changes the checksum."""
        X = torch.randn(64, 5)
        Y = X.clone()
        Y[13, 2] += 1.0
        assert _tensor_checksum(Y) != _tensor_checksum(X)

    def test_survives_copy_and_noncontiguity(self):
        """The checksum describes values, not memory layout."""
        X = torch.randn(64, 5)
        assert _tensor_checksum(X.clone()) == _tensor_checksum(X)
        assert _tensor_checksum(X.t().t()) == _tensor_checksum(X)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.float32, torch.float64, torch.int32]
    )
    def test_dtypes(self, dtype):
        """Every element size TorchDR accepts produces a usable checksum."""
        X = (torch.randn(64, 5) * 10).to(dtype)
        assert _tensor_checksum(X) != -1
        assert _tensor_checksum(X) == _tensor_checksum(X.clone())

    def test_empty(self):
        """An empty tensor is not an error."""
        assert _tensor_checksum(torch.empty(0, 5)) == 0

    def test_subsampled_rows_still_order_sensitive(self):
        """Above the row cap, a permutation is still detected."""
        n = contract._DIGEST_MAX_ROWS * 3
        X = torch.randn(n, 4)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
        assert _tensor_checksum(X[perm]) != _tensor_checksum(X)


class TestReplicatedInputAccepted:
    """Inputs that honour the contract must pass silently."""

    def test_identical_tensors(self, simulate_world):
        """The supported case: same full tensor on every rank."""
        X = torch.randn(128, 6)
        assert simulate_world([X, X, X, X]) is None

    def test_equal_but_distinct_tensors(self, simulate_world):
        """Equal content in separate allocations is still valid."""
        X = torch.randn(128, 6)
        assert simulate_world([X, X.clone(), X.clone()]) is None

    @pytest.mark.parametrize("n_samples", [1, 2, 4097])
    def test_degenerate_and_uneven_sizes(self, simulate_world, n_samples):
        """Sizes that do not divide evenly across ranks are not violations."""
        X = torch.randn(n_samples, 3)
        assert simulate_world([X, X, X, X]) is None

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_dtypes_accepted(self, simulate_world, dtype):
        """Any supported dtype passes as long as all ranks agree."""
        X = torch.randn(64, 4).to(dtype)
        assert simulate_world([X, X]) is None

    def test_replicated_dataloader(self, simulate_world):
        """A plain DataLoader over the full dataset is valid."""
        X = torch.randn(64, 4)
        assert simulate_world([_loader(X), _loader(X)]) is None

    @pytest.mark.parametrize("batch_size", [1, 7, 64, 1000])
    def test_uneven_batches(self, simulate_world, batch_size):
        """Batch size is a loader detail and must not affect the verdict."""
        X = torch.randn(64, 4)
        loaders = [_loader(X, batch_size=batch_size) for _ in range(2)]
        assert simulate_world(loaders) is None

    def test_differing_batch_sizes_across_ranks(self, simulate_world):
        """Ranks may batch differently as long as they iterate the same data."""
        X = torch.randn(64, 4)
        assert simulate_world([_loader(X, batch_size=8), _loader(X, 13)]) is None

    def test_duplicate_samples(self, simulate_world):
        """Repeated rows are legitimate data, not a contract violation."""
        X = torch.randn(32, 4).repeat(4, 1)
        assert simulate_world([X, X, X, X]) is None
        assert simulate_world([_loader(X), _loader(X)]) is None

    def test_all_rows_identical(self, simulate_world):
        """The degenerate duplicate case: every row the same."""
        X = torch.randn(1, 4).expand(256, 4).contiguous()
        assert simulate_world([X, X]) is None

    def test_no_op_without_distributed_context(self):
        """Non-distributed callers are unaffected."""
        X = torch.randn(32, 4)
        assert validate_distributed_input(X, None) is None
        assert validate_distributed_input(X, DistributedContext()) is None

    def test_repeated_calls_never_false_positive(self, simulate_world):
        """Validation is stateless; calling it repeatedly stays quiet."""
        X = torch.randn(256, 4)
        for _ in range(10):
            assert simulate_world([X, X]) is None


class TestViolationsDetected:
    """Each silent-corruption mode must become an early, actionable error."""

    def test_distributed_sampler_rejected(self, simulate_world):
        """The headline case: DistributedSampler(shuffle=False).

        A deterministic DistributedSampler passes TorchDR's determinism check,
        so before this contract each rank indexed only its own shard and
        returned local shard positions as if they were global sample ids.
        """
        X = torch.randn(64, 4)
        loaders = [
            _loader(
                X,
                sampler=DistributedSampler(
                    TensorDataset(X), num_replicas=2, rank=r, shuffle=False
                ),
            )
            for r in range(2)
        ]
        with pytest.raises(ValueError, match="DistributedSampler"):
            simulate_world(loaders)

    def test_distributed_sampler_error_names_the_shortfall(self, simulate_world):
        """The message states how many samples the rank actually sees."""
        X = torch.randn(64, 4)
        loaders = [
            _loader(
                X,
                sampler=DistributedSampler(
                    TensorDataset(X), num_replicas=2, rank=r, shuffle=False
                ),
            )
            for r in range(2)
        ]
        with pytest.raises(ValueError, match=r"yields 32 of the 64 samples"):
            simulate_world(loaders)

    def test_raises_on_every_rank(self, simulate_world):
        """A violation on one rank must not leave the others in a collective.

        If only the offending rank raised, the innocent ranks would proceed to
        the next collective and hang until the job timed out.
        """
        X = torch.randn(64, 4)
        loaders = [
            _loader(X),
            _loader(
                X,
                sampler=DistributedSampler(
                    TensorDataset(X), num_replicas=2, rank=1, shuffle=False
                ),
            ),
        ]
        for rank in (0, 1):
            with pytest.raises(ValueError, match="full dataset"):
                simulate_world(loaders, rank=rank)

    def test_mismatched_n_samples(self, simulate_world):
        """Ranks holding different row counts are rejected."""
        X = torch.randn(256, 4)
        with pytest.raises(ValueError, match="n_samples"):
            simulate_world([X, X[:137]])

    def test_mismatched_n_features(self, simulate_world):
        """Ranks holding different column counts are rejected."""
        X = torch.randn(256, 8)
        with pytest.raises(ValueError, match="n_features"):
            simulate_world([X, X[:, :7]])

    def test_mismatched_dtype(self, simulate_world):
        """Differing dtypes are rejected even though FAISS would cast them."""
        X = torch.randn(256, 4)
        with pytest.raises(ValueError, match="dtype"):
            simulate_world([X, X.double()])

    def test_permuted_rows(self, simulate_world):
        """Same shape and dtype, different row order: caught by the checksum."""
        X = torch.randn(256, 4)
        perm = torch.randperm(256, generator=torch.Generator().manual_seed(7))
        with pytest.raises(ValueError, match="not on content"):
            simulate_world([X, X[perm]])

    def test_different_data(self, simulate_world):
        """Same shape and dtype, unrelated values."""
        X = torch.randn(256, 4)
        Y = torch.randn(256, 4)
        with pytest.raises(ValueError, match="not on content"):
            simulate_world([X, Y])

    def test_mismatched_input_kind(self, simulate_world):
        """A tensor on one rank and a DataLoader on another is rejected."""
        X = torch.randn(64, 4)
        with pytest.raises(ValueError, match="input_kind"):
            simulate_world([X, _loader(X)])

    def test_dataloader_row_count_mismatch(self, simulate_world):
        """DataLoaders over differently sized datasets are rejected."""
        X = torch.randn(64, 4)
        with pytest.raises(ValueError, match="n_samples"):
            simulate_world([_loader(X), _loader(X[:48])])

    def test_error_mentions_the_offending_rank(self, simulate_world):
        """The message points at a specific rank, not just "some rank"."""
        X = torch.randn(64, 4)
        with pytest.raises(ValueError, match=r"rank 0 reports .* rank 2 reports"):
            simulate_world([X, X, X[:32], X])

    def test_sharded_loader_rejected_without_a_process_group(self):
        """world_size < 2 still rejects a sharded loader, with no collective.

        ``force_enable`` is used for single-process testing of distributed
        code paths, so this branch must not call into ``torch.distributed``.
        """
        X = torch.randn(64, 4)
        loader = _loader(
            X,
            sampler=DistributedSampler(
                TensorDataset(X), num_replicas=2, rank=0, shuffle=False
            ),
        )
        with pytest.raises(ValueError, match="DistributedSampler"):
            validate_distributed_input(loader, _context(0, 1))


class TestContentVerificationOptOut:
    """``verify_content=False`` trades content checking for a smaller cost."""

    def test_permutation_not_detected(self, simulate_world):
        """Without the checksum, a permutation is indistinguishable."""
        X = torch.randn(256, 4)
        perm = torch.randperm(256, generator=torch.Generator().manual_seed(7))
        assert simulate_world([X, X[perm]], verify_content=False) is None

    def test_shape_still_checked(self, simulate_world):
        """Metadata checks are unaffected by the opt-out."""
        X = torch.randn(256, 4)
        with pytest.raises(ValueError, match="n_samples"):
            simulate_world([X, X[:100]], verify_content=False)


class TestUnknownFieldsAreNotViolations:
    """A field a rank cannot measure must not be reported as a disagreement."""

    def test_unknown_checksum_is_tolerated(self, simulate_world, monkeypatch):
        """A rank that cannot checksum its input does not trip the check."""
        X = torch.randn(64, 4)
        calls = {"n": 0}
        real = contract._tensor_checksum

        def flaky(tensor):
            calls["n"] += 1
            return contract._UNKNOWN if calls["n"] > 1 else real(tensor)

        monkeypatch.setattr(contract, "_tensor_checksum", flaky)
        assert simulate_world([X, X]) is None
