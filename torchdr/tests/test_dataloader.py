"""Tests for DataLoader support in distance computation."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch
from torch.testing import assert_close
from torch.utils.data import DataLoader, TensorDataset

from torchdr.distance import (
    pairwise_distances,
    pairwise_distances_faiss_from_dataloader,
    FaissConfig,
)
from torchdr.distance.faiss import get_dataloader_metadata
from torchdr.utils import faiss


# Skip all tests if faiss is not available (faiss is False or None when not installed)
pytestmark = pytest.mark.skipif(
    faiss is None or faiss is False, reason="faiss not installed"
)


class TestDataLoaderDistances:
    """Test pairwise_distances with DataLoader input."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        n_samples = 1000
        n_features = 32
        X = torch.randn(n_samples, n_features)
        return X

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=100, shuffle=False)

    def test_dataloader_basic(self, sample_data, dataloader):
        """Test that DataLoader produces same results as tensor input."""
        k = 10

        # Compute with tensor
        dist_tensor, idx_tensor = pairwise_distances(
            sample_data, k=k, backend="faiss", return_indices=True
        )

        # Compute with DataLoader
        dist_dl, idx_dl = pairwise_distances(dataloader, k=k, return_indices=True)

        # Results should match
        assert_close(dist_tensor, dist_dl, rtol=1e-4, atol=1e-4)
        assert torch.equal(idx_tensor, idx_dl)

    def test_dataloader_exclude_diag(self, sample_data, dataloader):
        """Test exclude_diag with DataLoader."""
        k = 10

        # Compute with tensor
        dist_tensor, idx_tensor = pairwise_distances(
            sample_data, k=k, backend="faiss", exclude_diag=True, return_indices=True
        )

        # Compute with DataLoader
        dist_dl, idx_dl = pairwise_distances(
            dataloader, k=k, exclude_diag=True, return_indices=True
        )

        # Results should match
        assert_close(dist_tensor, dist_dl, rtol=1e-4, atol=1e-4)
        assert torch.equal(idx_tensor, idx_dl)

    @pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean"])
    def test_dataloader_metrics(self, sample_data, dataloader, metric):
        """Test different metrics with DataLoader."""
        k = 10

        # Compute with tensor
        dist_tensor, idx_tensor = pairwise_distances(
            sample_data, k=k, metric=metric, backend="faiss", return_indices=True
        )

        # Compute with DataLoader
        dist_dl, idx_dl = pairwise_distances(
            dataloader, k=k, metric=metric, return_indices=True
        )

        # Results should match
        assert_close(dist_tensor, dist_dl, rtol=1e-4, atol=1e-4)

    def test_dataloader_with_config(self, sample_data, dataloader):
        """Test DataLoader with FaissConfig."""
        k = 10
        config = FaissConfig(index_type="Flat")

        # Compute with DataLoader and config
        dist_dl, idx_dl = pairwise_distances(
            dataloader, k=k, backend=config, return_indices=True
        )

        # Should produce valid results
        assert dist_dl.shape == (len(sample_data), k)
        assert idx_dl.shape == (len(sample_data), k)

    def test_dataloader_requires_k(self, dataloader):
        """Test that DataLoader requires k parameter."""
        with pytest.raises(ValueError, match="k cannot be None"):
            pairwise_distances(dataloader, k=None)

    def test_dataloader_no_cross_distance(self, sample_data, dataloader):
        """Test that DataLoader doesn't support cross-distance."""
        Y = torch.randn(100, 32)
        with pytest.raises(ValueError, match="Y must be None"):
            pairwise_distances(dataloader, Y=Y, k=10)

    def test_dataloader_unsupported_backend(self, dataloader):
        """Test that DataLoader raises error for non-FAISS backends."""
        with pytest.raises(ValueError, match="only supports FAISS backend"):
            pairwise_distances(dataloader, k=10, backend="keops")

    def test_dataloader_different_batch_sizes(self, sample_data):
        """Test DataLoader with different batch sizes produces same results."""
        k = 10

        # Reference with full tensor
        dist_ref, idx_ref = pairwise_distances(
            sample_data, k=k, backend="faiss", return_indices=True
        )

        # Test different batch sizes
        for batch_size in [50, 100, 256, 500]:
            dataset = TensorDataset(sample_data)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            dist_dl, idx_dl = pairwise_distances(dl, k=k, return_indices=True)

            # Use looser tolerance due to float32 precision differences
            assert_close(dist_ref, dist_dl, rtol=1e-3, atol=1e-2)
            assert torch.equal(idx_ref, idx_dl)


class TestDataLoaderDirectFunction:
    """Test pairwise_distances_faiss_from_dataloader directly."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(500, 16)

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=50, shuffle=False)

    def test_direct_function_basic(self, sample_data, dataloader):
        """Test direct function call."""
        k = 5

        dist, idx = pairwise_distances_faiss_from_dataloader(
            dataloader, k=k, metric="sqeuclidean"
        )

        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)

    def test_direct_function_angular(self, sample_data):
        """Test angular metric."""
        # Normalize for angular metric
        sample_data = sample_data / sample_data.norm(dim=1, keepdim=True)
        dataset = TensorDataset(sample_data)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=False)

        k = 5
        dist, idx = pairwise_distances_faiss_from_dataloader(
            dataloader, k=k, metric="angular"
        )

        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDataLoaderGPU:
    """Test DataLoader with GPU computation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(500, 32)

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=100, shuffle=False)

    def test_dataloader_cuda(self, sample_data, dataloader):
        """Test DataLoader with CUDA device."""
        k = 10

        # Compute on CPU first for reference
        dist_cpu, idx_cpu = pairwise_distances(
            sample_data, k=k, backend="faiss", device="cpu", return_indices=True
        )

        # Compute with DataLoader on GPU
        dist_gpu, idx_gpu = pairwise_distances(
            dataloader, k=k, device="cuda", return_indices=True
        )

        # Move GPU results to CPU for comparison
        dist_gpu = dist_gpu.cpu()
        idx_gpu = idx_gpu.cpu()

        assert_close(dist_cpu, dist_gpu, rtol=1e-4, atol=1e-4)
        assert torch.equal(idx_cpu, idx_gpu)


class TestDataLoaderOptimizations:
    """Test DataLoader optimizations (metadata caching, shuffle validation)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(500, 32)

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=100, shuffle=False)

    def test_metadata_caching(self, sample_data, dataloader):
        """Test that metadata is cached after first pairwise_distances call."""
        k = 10

        # Before computation, metadata should not exist
        metadata_before = get_dataloader_metadata(dataloader)
        assert metadata_before is None

        # Compute k-NN (this should cache metadata)
        dist, idx = pairwise_distances(dataloader, k=k, return_indices=True)

        # After computation, metadata should be cached
        metadata_after = get_dataloader_metadata(dataloader)
        assert metadata_after is not None
        assert metadata_after["n_samples"] == len(sample_data)
        assert metadata_after["n_features"] == sample_data.shape[1]
        assert metadata_after["dtype"] == sample_data.dtype

        # Verify results are correct
        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)

    def test_shuffle_validation_sequential_sampler(self, sample_data):
        """Test that sequential sampler (shuffle=False) is accepted."""
        dataset = TensorDataset(sample_data)
        dl = DataLoader(dataset, batch_size=100, shuffle=False)

        # Should work without error
        dist, idx = pairwise_distances(dl, k=10, return_indices=True)
        assert dist.shape == (len(sample_data), 10)

    def test_shuffle_validation_random_sampler(self, sample_data):
        """Test that random sampler (shuffle=True) is rejected."""
        dataset = TensorDataset(sample_data)
        dl = DataLoader(dataset, batch_size=100, shuffle=True)

        # Should raise error about shuffle=True
        with pytest.raises(ValueError, match="shuffle=False"):
            pairwise_distances(dl, k=10, return_indices=True)

    def test_metadata_persists_across_calls(self, sample_data, dataloader):
        """Test that metadata persists and can be reused across multiple calls."""
        k = 10

        # First call caches metadata
        dist1, idx1 = pairwise_distances(dataloader, k=k, return_indices=True)

        # Get cached metadata
        metadata1 = get_dataloader_metadata(dataloader)
        assert metadata1 is not None
        assert metadata1["n_samples"] == len(sample_data)

        # Second call should still have metadata available
        dist2, idx2 = pairwise_distances(dataloader, k=k, return_indices=True)

        # Metadata should still be available after second call
        metadata2 = get_dataloader_metadata(dataloader)
        assert metadata2 is not None
        assert metadata2["n_samples"] == metadata1["n_samples"]
        assert metadata2["n_features"] == metadata1["n_features"]
        assert metadata2["dtype"] == metadata1["dtype"]

        # Results should be identical
        assert_close(dist1, dist2)
        assert torch.equal(idx1, idx2)

    def test_direct_function_caches_metadata(self, sample_data, dataloader):
        """Test that direct function also caches metadata."""
        k = 10

        # Call direct function
        dist, idx = pairwise_distances_faiss_from_dataloader(
            dataloader, k=k, metric="sqeuclidean"
        )

        # Verify metadata was cached
        metadata = get_dataloader_metadata(dataloader)
        assert metadata is not None
        assert metadata["n_samples"] == len(sample_data)
        assert metadata["n_features"] == sample_data.shape[1]


class TestAffinityMatcherDataLoader:
    """Test AffinityMatcher with DataLoader input."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        n_samples = 200
        n_features = 16
        X = torch.randn(n_samples, n_features)
        return X

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=50, shuffle=False)

    def test_affinity_matcher_dataloader_metadata(self, sample_data, dataloader):
        """Test that AffinityMatcher correctly extracts metadata from DataLoader."""
        from torchdr.affinity import UMAPAffinity, NormalizedStudentAffinity
        from torchdr.affinity_matcher import AffinityMatcher

        # Create AffinityMatcher with affinity that uses FAISS for k-NN
        model = AffinityMatcher(
            affinity_in=UMAPAffinity(n_neighbors=10, backend="faiss"),
            affinity_out=NormalizedStudentAffinity(),
            n_components=2,
            max_iter=2,  # Just enough to test the flow
            init="random",
            verbose=False,
        )

        # Fit with DataLoader
        embedding = model.fit_transform(dataloader)

        # Verify metadata was extracted correctly
        assert model.n_samples_in_ == len(sample_data)
        assert model.n_features_in_ == sample_data.shape[1]
        assert embedding.shape == (len(sample_data), 2)

    def test_affinity_matcher_dataloader_pca_init(self, sample_data, dataloader):
        """Test that AffinityMatcher uses IncrementalPCA for DataLoader + PCA init."""
        from torchdr.affinity import UMAPAffinity, NormalizedStudentAffinity
        from torchdr.affinity_matcher import AffinityMatcher

        model = AffinityMatcher(
            affinity_in=UMAPAffinity(n_neighbors=10, backend="faiss"),
            affinity_out=NormalizedStudentAffinity(),
            n_components=2,
            max_iter=2,
            init="pca",  # Should use IncrementalPCA for DataLoader
            verbose=False,
        )

        # Fit with DataLoader - should use IncrementalPCA internally
        embedding = model.fit_transform(dataloader)

        assert embedding.shape == (len(sample_data), 2)
