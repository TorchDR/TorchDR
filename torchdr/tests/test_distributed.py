"""Tests for distributed training utilities."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from torchdr.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
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
