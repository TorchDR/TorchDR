"""Tests for incremental PCA."""

# Author: @sirluk
#
# License: BSD 3-Clause License

import pytest
import torch
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.decomposition import IncrementalPCA as SkIncrementalPCA
from sklearn.model_selection import train_test_split
from torch.testing import assert_close

from torchdr import IncrementalPCA, ExactIncrementalPCA, PCA

torch.manual_seed(1999)

iris = datasets.load_iris()


def test_incremental_pca():
    # Incremental PCA on dense arrays.
    n_components = 2
    X = torch.tensor(iris.data, dtype=torch.float32)
    batch_size = X.shape[0] // 3
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(X)
    X_transformed = ipca.transform(X)

    # PCA
    U, S, Vh = torch.linalg.svd(X - torch.mean(X, dim=0))
    max_abs_rows = torch.argmax(torch.abs(Vh), dim=1)
    signs = torch.sign(Vh[range(Vh.shape[0]), max_abs_rows])
    Vh *= signs.view(-1, 1)
    explained_variance = S**2 / (X.size(0) - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    assert X_transformed.shape == (X.shape[0], 2)
    assert_close(
        ipca.explained_variance_ratio_.sum().item(),
        explained_variance_ratio[:n_components].sum().item(),
        rtol=1e-3,
        atol=1e-3,
    )


def test_incremental_pca_check_projection():
    # Test that the projection of data is correct.
    n, p = 100, 3
    X = torch.randn(n, p, dtype=torch.float64) * 0.1
    X[:10] += torch.tensor([3, 4, 5])
    Xt = 0.1 * torch.randn(1, p, dtype=torch.float64) + torch.tensor([3, 4, 5])

    # Get the reconstruction of the generated data X
    # Note that Xt has the same "components" as X, just separated
    # This is what we want to ensure is recreated correctly
    Yt = IncrementalPCA(n_components=2).fit(X).transform(Xt)

    # Normalize
    Yt /= torch.sqrt((Yt**2).sum())

    # Make sure that the first element of Yt is ~1, this means
    # the reconstruction worked as expected
    assert_close(torch.abs(Yt[0][0]).item(), 1.0, atol=1e-1, rtol=1e-1)


def test_incremental_pca_validation():
    # Test that n_components is <= n_features.
    X = torch.tensor([[0, 1, 0], [1, 0, 0]])
    n_samples, n_features = X.shape
    n_components = 4
    with pytest.raises((ValueError, AssertionError)):
        IncrementalPCA(n_components, batch_size=10).fit(X)

    # Tests that n_components is also <= n_samples.
    n_components = 3
    with pytest.raises((ValueError, AssertionError)):
        IncrementalPCA(n_components=n_components).partial_fit(X)


def test_n_componentsnone():
    # Ensures that n_components == None is handled correctly
    for n_samples, n_features in [(50, 10), (10, 50)]:
        X = torch.rand(n_samples, n_features)
        ipca = IncrementalPCA(n_components=None)

        # First partial_fit call, ipca.n_components is inferred from
        # min(X.shape)
        ipca.partial_fit(X)
        assert ipca.n_components == min(X.shape)


def test_incremental_pca_num_features_change():
    # Test that changing n_components will raise an error.
    n_samples = 100
    X = torch.randn(n_samples, 20)
    X2 = torch.randn(n_samples, 50)
    ipca = IncrementalPCA(n_components=None)
    ipca.fit(X)
    with pytest.raises(ValueError):
        ipca.partial_fit(X2)


def test_incremental_pca_batch_signs():
    # Test that components_ sign is stable over batch sizes.
    n_samples = 100
    n_features = 3
    X = torch.randn(n_samples, n_features)
    all_components = []
    batch_sizes = torch.arange(10, 20)
    for batch_size in batch_sizes:
        ipca = IncrementalPCA(n_components=None, batch_size=batch_size).fit(X)
        all_components.append(ipca.components_)

    for i, j in zip(all_components[:-1], all_components[1:]):
        assert_close(torch.sign(i), torch.sign(j), rtol=1e-6, atol=1e-6)


def test_incremental_pca_batch_values():
    # Test that components_ values are stable over batch sizes.
    n_samples = 100
    n_features = 3
    X = torch.randn(n_samples, n_features)
    all_components = []
    batch_sizes = torch.arange(20, 40, 3)
    for batch_size in batch_sizes:
        ipca = IncrementalPCA(n_components=None, batch_size=batch_size).fit(X)
        all_components.append(ipca.components_)

    for i, j in zip(all_components[:-1], all_components[1:]):
        assert_close(i, j, rtol=1e-1, atol=1e-1)


def test_incremental_pca_partial_fit():
    # Test that fit and partial_fit get equivalent results.
    n, p = 50, 3
    X = torch.randn(n, p)  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += torch.tensor([5, 4, 3])  # make a large mean

    # same check that we can find the original data from the transformed
    # signal (since the data is almost of rank n_components)
    batch_size = 10
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size).fit(X)
    pipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    # Add one to make sure endpoint is included
    batch_itr = torch.arange(0, n + 1, batch_size)
    for i, j in zip(batch_itr[:-1], batch_itr[1:]):
        pipca.partial_fit(X[i:j, :])
    assert_close(ipca.components_, pipca.components_, rtol=1e-3, atol=1e-3)


def test_incremental_pca_lowrank():
    # Test that lowrank mode is equivalent to non-lowrank mode.
    n_components = 2
    X = torch.tensor(iris.data, dtype=torch.float32)
    batch_size = X.shape[0] // 3

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(X)

    ipcalr = IncrementalPCA(
        n_components=n_components, batch_size=batch_size, lowrank=True
    )
    ipcalr.fit(X)

    assert_close(ipca.components_, ipcalr.components_, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("n_components", [10, 20, None])
def test_incremental_pca_on_digits(n_components):
    """
    Test that our custom IncrementalPCA produces similar results to sklearn's
    IncrementalPCA on the digits dataset, both in partial_fit and fit/transform usage.
    """

    X, y = load_digits(return_X_y=True)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    ipca = IncrementalPCA(
        n_components=n_components,
        batch_size=64,
    )
    sklearn_ipca = SkIncrementalPCA(
        n_components=n_components,
        batch_size=64,
    )

    ipca.fit(X_train)
    sklearn_ipca.fit(X_train)

    X_train_fit = ipca.transform(X_train)
    X_train_sklearn_fit = sklearn_ipca.transform(X_train)

    X_test_fit = ipca.transform(X_test)
    X_test_sklearn_fit = sklearn_ipca.transform(X_test)

    assert_allclose(X_train_sklearn_fit, X_train_fit, rtol=1e-5, atol=1e-5)
    assert_allclose(X_test_sklearn_fit, X_test_fit, rtol=1e-5, atol=1e-5)


# ================ Tests for ExactIncrementalPCA ================


def test_exact_incremental_pca_vs_pca():
    """Test that ExactIncrementalPCA gives same results as regular PCA."""
    n_components = 2
    X = torch.tensor(iris.data, dtype=torch.float32)

    # Regular PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)

    # ExactIncrementalPCA with single batch
    exact_ipca = ExactIncrementalPCA(n_components=n_components)
    exact_ipca.compute_mean([X])
    exact_ipca.fit([X])
    X_exact = exact_ipca.transform(X)

    # Compare reconstructions (components might have different signs)
    reconstruction_pca = X_pca @ pca.components_ + pca.mean_
    reconstruction_exact = X_exact @ exact_ipca.components_ + exact_ipca.mean_

    # Check reconstructions are similar (allow more tolerance due to different algorithms)
    assert_close(reconstruction_pca, reconstruction_exact, rtol=1e-2, atol=1e-2)

    # Check explained variance is in descending order
    assert torch.all(
        exact_ipca.explained_variance_[:-1] >= exact_ipca.explained_variance_[1:]
    ), "Explained variance should be in descending order"


def test_exact_incremental_pca_batches():
    """Test that batched processing gives same results as single batch."""
    n_components = 2
    X = torch.tensor(iris.data, dtype=torch.float32)
    batch_size = X.shape[0] // 3

    # Single batch
    exact_ipca_single = ExactIncrementalPCA(n_components=n_components)
    exact_ipca_single.compute_mean([X])
    exact_ipca_single.fit([X])
    X_single = exact_ipca_single.transform(X)

    # Multiple batches
    batches = [X[i : i + batch_size] for i in range(0, X.shape[0], batch_size)]
    exact_ipca_batched = ExactIncrementalPCA(n_components=n_components)
    exact_ipca_batched.compute_mean(batches)
    exact_ipca_batched.fit(batches)
    X_batched = exact_ipca_batched.transform(X)

    # Compare reconstructions
    reconstruction_single = (
        X_single @ exact_ipca_single.components_ + exact_ipca_single.mean_
    )
    reconstruction_batched = (
        X_batched @ exact_ipca_batched.components_ + exact_ipca_batched.mean_
    )

    assert_close(reconstruction_single, reconstruction_batched, rtol=1e-5, atol=1e-5)


def test_exact_incremental_pca_check_projection():
    """Test that the projection of data is correct."""
    n, p = 100, 3
    X = torch.randn(n, p, dtype=torch.float64) * 0.1
    X[:10] += torch.tensor([3, 4, 5])
    Xt = 0.1 * torch.randn(1, p, dtype=torch.float64) + torch.tensor([3, 4, 5])

    # Get the reconstruction
    exact_ipca = ExactIncrementalPCA(n_components=2)
    exact_ipca.compute_mean([X])
    exact_ipca.fit([X])
    Yt = exact_ipca.transform(Xt)

    # Normalize
    Yt /= torch.sqrt((Yt**2).sum())

    # Make sure that the first element of Yt is ~1
    assert_close(torch.abs(Yt[0][0]).item(), 1.0, atol=1e-1, rtol=1e-1)


def test_exact_incremental_pca_validation():
    """Test validation of n_components."""
    X = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
    n_features = X.shape[1]

    # Test that n_components <= n_features
    n_components = 4
    exact_ipca = ExactIncrementalPCA(n_components=n_components)
    exact_ipca.compute_mean([X])
    # Should work since we clamp n_components internally
    exact_ipca.fit([X])
    assert exact_ipca.components_.shape[0] <= n_features


def test_exact_incremental_pca_partial_fit():
    """Test partial_fit method."""
    X = torch.tensor(iris.data, dtype=torch.float32)
    n_components = 2
    batch_size = 50

    exact_ipca = ExactIncrementalPCA(n_components=n_components)

    # First compute mean (required)
    exact_ipca.compute_mean([X])

    # Then use partial_fit for each batch
    for i in range(0, X.shape[0], batch_size):
        batch = X[i : i + batch_size]
        exact_ipca.partial_fit(batch)

    # Compute components after all batches
    exact_ipca._compute_components()

    # Transform and check shape
    X_transformed = exact_ipca.transform(X)
    assert X_transformed.shape == (X.shape[0], n_components)


def test_exact_incremental_pca_fit_transform():
    """Test fit_transform method."""
    X = torch.tensor(iris.data, dtype=torch.float32)
    n_components = 2

    exact_ipca = ExactIncrementalPCA(n_components=n_components)
    X_transformed = exact_ipca.fit_transform(X)

    assert X_transformed.shape == (X.shape[0], n_components)

    # Check that fit_transform gives same result as fit then transform
    exact_ipca2 = ExactIncrementalPCA(n_components=n_components)
    exact_ipca2.compute_mean([X])
    exact_ipca2.fit([X])
    X_transformed2 = exact_ipca2.transform(X)

    # Compare reconstructions (components might have different signs)
    reconstruction1 = X_transformed @ exact_ipca.components_ + exact_ipca.mean_
    reconstruction2 = X_transformed2 @ exact_ipca2.components_ + exact_ipca2.mean_

    assert_close(reconstruction1, reconstruction2, rtol=1e-4, atol=1e-4)


def test_exact_incremental_pca_mean_not_computed():
    """Test error when mean is not computed before partial_fit."""
    X = torch.tensor(iris.data, dtype=torch.float32)
    exact_ipca = ExactIncrementalPCA(n_components=2)

    with pytest.raises(ValueError, match="Mean must be computed first"):
        exact_ipca.partial_fit(X)


def test_exact_incremental_pca_different_dtypes():
    """Test with different data types."""
    X32 = torch.tensor(iris.data, dtype=torch.float32)
    X64 = torch.tensor(iris.data, dtype=torch.float64)
    n_components = 2

    # Float32
    exact_ipca32 = ExactIncrementalPCA(n_components=n_components)
    exact_ipca32.compute_mean([X32])
    exact_ipca32.fit([X32])
    X_transformed32 = exact_ipca32.transform(X32)

    # Float64
    exact_ipca64 = ExactIncrementalPCA(n_components=n_components)
    exact_ipca64.compute_mean([X64])
    exact_ipca64.fit([X64])
    X_transformed64 = exact_ipca64.transform(X64)

    # Check shapes
    assert X_transformed32.shape == (X32.shape[0], n_components)
    assert X_transformed64.shape == (X64.shape[0], n_components)

    # Results should be similar despite different precision
    assert_close(X_transformed32.double(), X_transformed64, rtol=1e-4, atol=1e-4)


# ================ Tests for DataLoader support ================


def test_incremental_pca_dataloader():
    """Test that IncrementalPCA works with DataLoader input."""
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(iris.data, dtype=torch.float32)
    n_components = 2
    batch_size = 50

    # Create DataLoader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Fit with DataLoader (disable process_duplicates for fair comparison)
    ipca_dl = IncrementalPCA(n_components=n_components, process_duplicates=False)
    X_transformed_dl = ipca_dl.fit_transform(dataloader)

    # Fit with tensor using same batch size for comparison
    ipca_tensor = IncrementalPCA(
        n_components=n_components, batch_size=batch_size, process_duplicates=False
    )
    ipca_tensor.fit_transform(X)

    # The learned components should be the same
    assert_close(ipca_dl.components_, ipca_tensor.components_, rtol=1e-5, atol=1e-5)
    assert_close(ipca_dl.mean_, ipca_tensor.mean_, rtol=1e-5, atol=1e-5)

    # Verify DataLoader result is correct by checking reconstruction
    # (manual transform on fresh data to avoid in-place modification issue in tensor path)
    X_fresh = torch.tensor(iris.data, dtype=torch.float32)
    X_expected = (X_fresh - ipca_dl.mean_) @ ipca_dl.components_.T
    assert X_transformed_dl.shape == X_expected.shape
    assert_close(X_transformed_dl, X_expected.float(), rtol=1e-4, atol=1e-4)


def test_exact_incremental_pca_dataloader():
    """Test that ExactIncrementalPCA works with DataLoader input."""
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(iris.data, dtype=torch.float32)
    n_components = 2

    # Create DataLoader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)

    # Fit with DataLoader (use fit() then transform() since fit_transform doesn't
    # support DataLoader)
    exact_ipca_dl = ExactIncrementalPCA(n_components=n_components)
    exact_ipca_dl.compute_mean(dataloader)
    exact_ipca_dl.fit(dataloader)
    X_transformed_dl = exact_ipca_dl.transform(X)

    # Fit with tensor for comparison
    exact_ipca_tensor = ExactIncrementalPCA(n_components=n_components)
    X_transformed_tensor = exact_ipca_tensor.fit_transform(X)

    # Results should be similar (reconstruction-wise)
    reconstruction_dl = (
        X_transformed_dl @ exact_ipca_dl.components_ + exact_ipca_dl.mean_
    )
    reconstruction_tensor = (
        X_transformed_tensor @ exact_ipca_tensor.components_ + exact_ipca_tensor.mean_
    )

    assert X_transformed_dl.shape == X_transformed_tensor.shape
    assert_close(reconstruction_dl, reconstruction_tensor, rtol=1e-4, atol=1e-4)


def test_exact_incremental_pca_dataloader_compute_mean():
    """Test that compute_mean works with DataLoader."""
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(iris.data, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)

    # Compute mean with DataLoader
    exact_ipca = ExactIncrementalPCA(n_components=2)
    exact_ipca.compute_mean(dataloader)

    # Compare to tensor mean
    expected_mean = X.mean(dim=0)
    assert_close(exact_ipca.mean_, expected_mean, rtol=1e-5, atol=1e-5)


def test_exact_incremental_pca_dataloader_fit():
    """Test that fit works with DataLoader."""
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(iris.data, dtype=torch.float32)
    n_components = 2

    # Create DataLoader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)

    # Fit with DataLoader
    exact_ipca = ExactIncrementalPCA(n_components=n_components)
    exact_ipca.compute_mean(dataloader)
    exact_ipca.fit(dataloader)

    # Check components shape
    assert exact_ipca.components_.shape == (n_components, X.shape[1])
