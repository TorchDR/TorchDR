"""Tests for Distributed PCA."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch
from torch.testing import assert_close
from sklearn import datasets
from unittest.mock import patch

from torchdr import DistributedPCA, PCA

torch.manual_seed(42)

iris = datasets.load_iris()


class TestDistributedPCANonDistributed:
    """Tests for DistributedPCA when torch.distributed is not initialized."""

    def test_fallback_to_non_distributed(self):
        """Test that DistributedPCA falls back to non-distributed mode."""
        X = torch.tensor(iris.data, dtype=torch.float32)
        n_components = 2

        # In non-distributed mode, DistributedPCA should work like regular PCA
        dpca = DistributedPCA(n_components=n_components)
        X_transformed = dpca.fit_transform(X)

        assert X_transformed.shape == (X.shape[0], n_components)
        assert dpca.components_.shape == (n_components, X.shape[1])
        assert dpca.mean_.shape == (1, X.shape[1])

    def test_matches_regular_pca(self):
        """Test that non-distributed DistributedPCA matches regular PCA."""
        X = torch.tensor(iris.data, dtype=torch.float32)
        n_components = 2

        dpca = DistributedPCA(n_components=n_components)
        X_dpca = dpca.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        # Compare reconstructions (components might have different signs)
        reconstruction_dpca = X_dpca @ dpca.components_ + dpca.mean_
        reconstruction_pca = X_pca @ pca.components_ + pca.mean_

        assert_close(reconstruction_dpca, reconstruction_pca, rtol=1e-4, atol=1e-4)

    def test_transform(self):
        """Test transform method."""
        X = torch.tensor(iris.data, dtype=torch.float32)
        n_components = 2

        dpca = DistributedPCA(n_components=n_components)
        dpca.fit(X)
        X_transformed = dpca.transform(X)

        assert X_transformed.shape == (X.shape[0], n_components)

    def test_transform_not_fitted(self):
        """Test that transform raises error when not fitted."""
        X = torch.tensor(iris.data, dtype=torch.float32)
        dpca = DistributedPCA(n_components=2)

        with pytest.raises(ValueError, match="not fitted"):
            dpca.transform(X)

    def test_numpy_input(self):
        """Test with numpy input."""
        import numpy as np

        X_np = iris.data.astype(np.float32)
        n_components = 2

        dpca = DistributedPCA(n_components=n_components)
        X_transformed = dpca.fit_transform(X_np)

        # Should return numpy array
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == (X_np.shape[0], n_components)

    def test_different_n_components(self):
        """Test with different numbers of components."""
        X = torch.tensor(iris.data, dtype=torch.float32)

        for n_components in [1, 2, 3, 4]:
            dpca = DistributedPCA(n_components=n_components)
            X_transformed = dpca.fit_transform(X)
            assert X_transformed.shape == (X.shape[0], n_components)

    def test_deterministic_output(self):
        """Test that output is deterministic."""
        X = torch.tensor(iris.data, dtype=torch.float32)
        n_components = 2

        dpca1 = DistributedPCA(n_components=n_components, random_state=42)
        X1 = dpca1.fit_transform(X)

        dpca2 = DistributedPCA(n_components=n_components, random_state=42)
        X2 = dpca2.fit_transform(X)

        assert_close(X1, X2)


class TestDistributedPCAMocked:
    """Tests for DistributedPCA with mocked distributed functions."""

    def test_distributed_computation_single_gpu(self):
        """Test distributed computation with single GPU (world_size=1)."""
        X = torch.tensor(iris.data, dtype=torch.float32)
        n_components = 2

        # Mock distributed environment with single GPU
        dist_pca_module = "torchdr.spectral_embedding.distributed_pca"
        with (
            patch(f"{dist_pca_module}.is_distributed", return_value=True),
            patch(f"{dist_pca_module}.get_rank", return_value=0),
            patch(f"{dist_pca_module}.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.all_reduce") as mock_all_reduce,
        ):
            # Make all_reduce a no-op (single GPU doesn't need actual communication)
            mock_all_reduce.side_effect = lambda tensor, op: None

            dpca = DistributedPCA(n_components=n_components)
            X_transformed = dpca.fit_transform(X)

            assert X_transformed.shape == (X.shape[0], n_components)
            assert dpca.components_.shape == (n_components, X.shape[1])

    def test_distributed_mean_computation(self):
        """Test that distributed mean computation is correct."""
        # Create two "chunks" of data
        X1 = torch.randn(50, 10)
        X2 = torch.randn(50, 10)
        X_full = torch.cat([X1, X2], dim=0)

        # Expected mean from full data
        expected_mean = X_full.mean(dim=0)

        # Simulate distributed mean computation
        local_sum_1 = X1.sum(dim=0)
        local_sum_2 = X2.sum(dim=0)
        global_sum = local_sum_1 + local_sum_2
        global_mean = global_sum / 100

        assert_close(global_mean, expected_mean, rtol=1e-5, atol=1e-5)

    def test_distributed_covariance_computation(self):
        """Test that distributed covariance computation is correct."""
        # Create two "chunks" of data
        X1 = torch.randn(50, 10)
        X2 = torch.randn(50, 10)
        X_full = torch.cat([X1, X2], dim=0)

        # Global mean
        global_mean = X_full.mean(dim=0, keepdim=True)

        # Expected covariance from full data
        X_centered_full = X_full - global_mean
        expected_cov = (X_centered_full.T @ X_centered_full) / 100

        # Simulate distributed covariance computation
        X1_centered = X1 - global_mean
        X2_centered = X2 - global_mean
        local_cov_1 = X1_centered.T @ X1_centered
        local_cov_2 = X2_centered.T @ X2_centered
        global_cov = (local_cov_1 + local_cov_2) / 100

        assert_close(global_cov, expected_cov, rtol=1e-5, atol=1e-5)


class TestDistributedPCAEdgeCases:
    """Tests for edge cases."""

    def test_few_samples(self):
        """Test with few samples (edge case)."""
        X = torch.randn(5, 10)
        dpca = DistributedPCA(n_components=2)
        X_transformed = dpca.fit_transform(X)
        assert X_transformed.shape == (5, 2)

    def test_n_components_equals_n_features(self):
        """Test when n_components equals n_features."""
        X = torch.randn(100, 5)
        dpca = DistributedPCA(n_components=5)
        X_transformed = dpca.fit_transform(X)
        assert X_transformed.shape == (100, 5)

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        X = torch.randn(50, 100)
        dpca = DistributedPCA(n_components=10)
        X_transformed = dpca.fit_transform(X)
        assert X_transformed.shape == (50, 10)

    def test_float64(self):
        """Test with float64 data."""
        X = torch.tensor(iris.data, dtype=torch.float64)
        dpca = DistributedPCA(n_components=2)
        X_transformed = dpca.fit_transform(X)
        assert X_transformed.dtype == torch.float64


class TestDistributedPCAReproducibility:
    """Tests for reproducibility."""

    def test_reconstruction_error(self):
        """Test that reconstruction error is reasonable."""
        X = torch.tensor(iris.data, dtype=torch.float32)
        n_components = 4  # Use all components for near-perfect reconstruction

        dpca = DistributedPCA(n_components=n_components)
        X_transformed = dpca.fit_transform(X)
        X_reconstructed = X_transformed @ dpca.components_ + dpca.mean_

        # Reconstruction error should be very small with all components
        error = torch.mean((X - X_reconstructed) ** 2)
        assert error < 0.1

    def test_variance_preservation(self):
        """Test that PCA preserves variance in order of components."""
        X = torch.tensor(iris.data, dtype=torch.float32)
        n_components = 4

        dpca = DistributedPCA(n_components=n_components)
        X_transformed = dpca.fit_transform(X)

        # Variance should decrease across components
        variances = X_transformed.var(dim=0)
        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i + 1] - 1e-5
