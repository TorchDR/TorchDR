"""Tests for distributed training utilities."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import warnings
from unittest.mock import patch

import pytest
import torch

import torchdr.distributed as torchdr_distributed
from torchdr.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    shutdown_distributed,
    DistributedContext,
)


class TestDistributedUtilities:
    """Tests for utility functions."""

    def test_is_distributed_false(self):
        """Test is_distributed returns False when not initialized."""
        # In test environment, distributed is not initialized
        assert is_distributed() is False

    def test_get_rank_non_distributed(self):
        """Test get_rank returns 0 when not distributed."""
        assert get_rank() == 0

    def test_get_world_size_non_distributed(self):
        """Test get_world_size returns 1 when not distributed."""
        assert get_world_size() == 1


class TestInitDistributed:
    """Tests for the authoritative process group entry point."""

    def test_no_launcher_is_noop(self, monkeypatch):
        """Without LOCAL_RANK there is no rendezvous, so nothing is created."""
        monkeypatch.delenv("LOCAL_RANK", raising=False)

        with patch("torch.distributed.init_process_group") as mock_init:
            assert init_distributed() is False

        mock_init.assert_not_called()

    def test_creates_group_when_launched(self, monkeypatch):
        """A launcher rendezvous triggers exactly one initialization."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(torchdr_distributed, "_cleanup_registered", True)
        monkeypatch.setattr(
            torchdr_distributed, "_distributed_initialized_by_torchdr", False
        )

        with patch("torch.distributed.init_process_group") as mock_init:
            with patch("torch.cuda.is_available", return_value=False):
                assert init_distributed() is True

        mock_init.assert_called_once_with(backend="gloo")
        assert torchdr_distributed._distributed_initialized_by_torchdr is True

        # Reset the module flag so later tests see a pristine state.
        monkeypatch.setattr(
            torchdr_distributed, "_distributed_initialized_by_torchdr", False
        )

    def test_defaults_to_nccl_with_cuda(self, monkeypatch):
        """CUDA launches use NCCL and bind the local device."""
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setattr(torchdr_distributed, "_cleanup_registered", True)
        monkeypatch.setattr(
            torchdr_distributed, "_distributed_initialized_by_torchdr", False
        )

        with patch("torch.distributed.init_process_group") as mock_init:
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.set_device") as mock_set_device:
                    assert init_distributed() is True

        mock_init.assert_called_once_with(backend="nccl")
        mock_set_device.assert_called_once_with(1)

        monkeypatch.setattr(
            torchdr_distributed, "_distributed_initialized_by_torchdr", False
        )

    def test_idempotent_when_group_exists(self, monkeypatch):
        """A second call must not raise 'initialize the default group twice'."""
        monkeypatch.setenv("LOCAL_RANK", "0")

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_backend", return_value="nccl"):
                with patch("torch.distributed.init_process_group") as mock_init:
                    assert init_distributed() is False

        mock_init.assert_not_called()

    def test_warns_on_backend_mismatch(self, monkeypatch):
        """An explicit backend that cannot be honoured is reported."""
        monkeypatch.setenv("LOCAL_RANK", "0")

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_backend", return_value="nccl"):
                with pytest.warns(UserWarning, match="already initialized"):
                    assert init_distributed(backend="gloo") is False

    def test_no_warning_when_backend_matches(self, monkeypatch):
        """Requesting the backend already in use is silent."""
        monkeypatch.setenv("LOCAL_RANK", "0")

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_backend", return_value="gloo"):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    assert init_distributed(backend="gloo") is False


class TestShutdownDistributed:
    """Tests for the matching teardown helper."""

    def test_noop_for_foreign_group(self, monkeypatch):
        """A group TorchDR did not create is left alone."""
        monkeypatch.setattr(
            torchdr_distributed, "_distributed_initialized_by_torchdr", False
        )

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                assert shutdown_distributed() is False

        mock_destroy.assert_not_called()

    def test_destroys_own_group_once(self, monkeypatch):
        """A group TorchDR created is destroyed exactly once."""
        monkeypatch.setattr(
            torchdr_distributed, "_distributed_initialized_by_torchdr", True
        )

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                assert shutdown_distributed() is True
                assert shutdown_distributed() is False

        mock_destroy.assert_called_once()


class TestDistributedContext:
    """Tests for DistributedContext class."""

    def test_init_non_distributed(self):
        """Test initialization when distributed is not active."""
        ctx = DistributedContext()

        assert ctx.is_initialized is False
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.local_rank == 0

    def test_repr_non_distributed(self):
        """Test string representation when not initialized."""
        ctx = DistributedContext()
        assert "not initialized" in repr(ctx)

    def test_force_enable(self):
        """Test force_enable flag."""
        ctx = DistributedContext(force_enable=True)
        assert ctx.is_initialized is True


class TestComputeChunkBounds:
    """Tests for compute_chunk_bounds method."""

    def test_even_division(self):
        """Test chunk bounds with evenly divisible samples."""
        # Simulate 4 GPUs with 100 samples
        for rank in range(4):
            ctx = DistributedContext()
            ctx.rank = rank
            ctx.world_size = 4

            start, end = ctx.compute_chunk_bounds(100)

            assert end - start == 25  # Each gets 25
            assert start == rank * 25

    def test_uneven_division(self):
        """Test chunk bounds with remainder."""
        # 97 samples across 4 GPUs: 25, 24, 24, 24
        chunk_sizes = []
        for rank in range(4):
            ctx = DistributedContext()
            ctx.rank = rank
            ctx.world_size = 4

            start, end = ctx.compute_chunk_bounds(97)
            chunk_sizes.append(end - start)

        # First rank gets extra sample
        assert chunk_sizes[0] == 25
        assert chunk_sizes[1:] == [24, 24, 24]
        assert sum(chunk_sizes) == 97

    def test_single_gpu(self):
        """Test with single GPU (world_size=1)."""
        ctx = DistributedContext()
        ctx.rank = 0
        ctx.world_size = 1

        start, end = ctx.compute_chunk_bounds(100)

        assert start == 0
        assert end == 100

    def test_more_gpus_than_samples(self):
        """Test edge case: more GPUs than samples."""
        # 3 samples across 5 GPUs
        chunk_sizes = []
        for rank in range(5):
            ctx = DistributedContext()
            ctx.rank = rank
            ctx.world_size = 5

            start, end = ctx.compute_chunk_bounds(3)
            chunk_sizes.append(end - start)

        # First 3 ranks get 1 sample each, rest get 0
        assert chunk_sizes == [1, 1, 1, 0, 0]

    def test_chunks_cover_all_samples(self):
        """Test that chunks fully cover the sample range without gaps."""
        ctx = DistributedContext()
        n_samples = 103
        world_size = 7

        all_indices = set()
        for rank in range(world_size):
            ctx.rank = rank
            ctx.world_size = world_size
            start, end = ctx.compute_chunk_bounds(n_samples)
            all_indices.update(range(start, end))

        assert all_indices == set(range(n_samples))


class TestGetRankForIndices:
    """Tests for get_rank_for_indices static method."""

    def test_basic(self):
        """Test basic rank assignment."""
        # 100 samples, 4 GPUs: each owns 25
        indices = torch.tensor([0, 24, 25, 49, 50, 74, 75, 99])
        ranks = DistributedContext.get_rank_for_indices(
            indices, n_samples=100, world_size=4
        )

        expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        assert torch.equal(ranks, expected)

    def test_uneven_distribution(self):
        """Test with uneven sample distribution."""
        # 10 samples, 3 GPUs: 4, 3, 3
        # Rank 0: [0,1,2,3], Rank 1: [4,5,6], Rank 2: [7,8,9]
        indices = torch.tensor([0, 3, 4, 6, 7, 9])
        ranks = DistributedContext.get_rank_for_indices(
            indices, n_samples=10, world_size=3
        )

        expected = torch.tensor([0, 0, 1, 1, 2, 2])
        assert torch.equal(ranks, expected)

    def test_inverse_of_compute_chunk_bounds(self):
        """Test that get_rank_for_indices is inverse of compute_chunk_bounds."""
        n_samples = 97
        world_size = 4

        for rank in range(world_size):
            ctx = DistributedContext()
            ctx.rank = rank
            ctx.world_size = world_size
            start, end = ctx.compute_chunk_bounds(n_samples)

            # All indices in this chunk should map to this rank
            chunk_indices = torch.arange(start, end)
            computed_ranks = DistributedContext.get_rank_for_indices(
                chunk_indices, n_samples, world_size
            )
            assert (computed_ranks == rank).all()


class TestGetFaissConfig:
    """Tests for get_faiss_config method."""

    def test_default_config(self):
        """Test creating default config."""
        ctx = DistributedContext()
        ctx.local_rank = 2

        config = ctx.get_faiss_config()

        assert config.device == 2

    def test_with_base_config(self):
        """Test creating config from base config."""
        from torchdr.distance import FaissConfig

        ctx = DistributedContext()
        ctx.local_rank = 3

        base = FaissConfig(temp_memory=4.0, index_type="IVF", nprobe=10)
        config = ctx.get_faiss_config(base)

        # Should copy settings but override device
        assert config.device == 3
        assert config.temp_memory == 4.0
        assert config.index_type == "IVF"
        assert config.nprobe == 10

    def test_with_ivfpq_base_config(self):
        """Test preserving product quantization settings from a base config."""
        from torchdr.distance import FaissConfig

        ctx = DistributedContext()
        ctx.local_rank = 2

        base = FaissConfig(
            temp_memory=1.5,
            device=7,
            index_type="IVFPQ",
            nprobe=12,
            nlist=256,
            M=32,
            nbits=6,
            useFloat16=True,
        )
        config = ctx.get_faiss_config(base)

        assert config.device == 2
        assert config.temp_memory == 1.5
        assert config.index_type == "IVFPQ"
        assert config.nprobe == 12
        assert config.nlist == 256
        assert config.M == 32
        assert config.nbits == 6
        assert config.faiss_kwargs == {"useFloat16": True}
