# Author: Mathurin Massias
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch
from sklearn.decomposition import KernelPCA as skKernelPCA

from torchdr.affinity import NormalizedGaussianAffinity, SinkhornAffinity
from torchdr.utils import pykeops

from torchdr import KernelPCA, PHATE, PCA, ExactIncrementalPCA

from torchdr.tests.utils import toy_dataset


DEVICES = ["cpu"]
USE_KEOPS = [False]
if pykeops:
    USE_KEOPS.append(True)


@pytest.mark.parametrize("n_components", [3, None])
def test_KernelPCA_sklearn(n_components):
    torch.manual_seed(0)
    X = torch.randn(10, 20)
    X /= torch.linalg.norm(X, axis=0, keepdims=True)
    # otherwise all points at distance 1
    sigma = 2
    aff = NormalizedGaussianAffinity(
        zero_diag=False, sigma=sigma, normalization_dim=None
    )
    model = KernelPCA(affinity=aff, n_components=n_components)
    rtol = 1e-2  # we might want to take a look at that someday

    # Test fit_transform
    res_1 = model.fit_transform(X)

    # Test that fit then fit_transform gives same result
    model.fit(X)
    res_2 = model.fit_transform(X)
    np.testing.assert_allclose(res_1, res_2, rtol=rtol, atol=1e-5)

    # Compare with sklearn for Gaussian kernel (fit_transform only)
    model_sk = skKernelPCA(
        n_components=n_components, kernel="rbf", gamma=1 / sigma
    ).fit(X)
    X_sk = model_sk.transform(X)
    # NB: signs can be opposite, so we test allclose on absolute values
    np.testing.assert_allclose(np.abs(X_sk), np.abs(res_1), rtol=rtol)


def test_KernelPCA_no_transform():
    torch.manual_seed(0)
    X = torch.randn(10, 20)
    X /= torch.linalg.norm(X, axis=0)
    n_components = 3
    aff = SinkhornAffinity(zero_diag=False)
    model = KernelPCA(affinity=aff, n_components=n_components)

    # Test that fit and fit_transform work
    model.fit(X)
    result = model.fit_transform(X)
    assert result.shape == (X.shape[0], n_components)


@pytest.mark.skipif(not pykeops, reason="pykeops is not available")
def test_KernelPCA_keops():
    with pytest.raises(NotImplementedError):
        KernelPCA(backend="keops")


@pytest.mark.parametrize("device", DEVICES)
def test_phate(device):
    torch.autograd.set_detect_anomaly(True)
    data, _ = toy_dataset()
    data = torch.tensor(data, dtype=torch.float32)
    data = data.to(device)
    # PHATE only supports backend=None since it uses NegPotentialAffinity
    phate = PHATE(backend=None, device=device)
    embedding = phate.fit_transform(data)
    assert embedding.shape == (data.shape[0], 2)

    # Test that faiss and keops backend raises an error from PHATE class
    with pytest.raises(ValueError):
        PHATE(backend="faiss", device=device)
        PHATE(backend="keops", device=device)


def test_exact_incremental_pca():
    """Test that ExactIncrementalPCA gives same results as regular PCA."""
    torch.manual_seed(42)
    n_samples = 500
    n_features = 20
    n_components = 5

    # Generate low-rank data
    true_components = torch.randn(n_components, n_features)
    coefficients = torch.randn(n_samples, n_components)
    X = coefficients @ true_components + 0.1 * torch.randn(n_samples, n_features)

    # Compute with regular PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Compute with ExactIncrementalPCA (single batch)
    exact_ipca = ExactIncrementalPCA(n_components=n_components)
    exact_ipca.compute_mean([X])
    exact_ipca.fit([X])
    X_exact = exact_ipca.transform(X)

    # Compare reconstructions (components might have different signs)
    reconstruction_pca = X_pca @ pca.components_ + pca.mean_
    reconstruction_exact = X_exact @ exact_ipca.components_ + exact_ipca.mean_

    # Check that reconstructions are similar
    reconstruction_error = torch.norm(
        reconstruction_pca - reconstruction_exact
    ) / torch.norm(X)
    assert reconstruction_error < 1e-4, (
        f"Reconstruction error too large: {reconstruction_error}"
    )

    # Test with multiple batches
    batch_size = 100
    batches = [X[i : i + batch_size] for i in range(0, n_samples, batch_size)]

    # Compute with ExactIncrementalPCA (multiple batches)
    exact_ipca_batched = ExactIncrementalPCA(n_components=n_components)
    exact_ipca_batched.compute_mean(batches)
    exact_ipca_batched.fit(batches)
    X_exact_batched = exact_ipca_batched.transform(X)

    # Check that batched version gives same result as single batch
    reconstruction_exact_batched = (
        X_exact_batched @ exact_ipca_batched.components_ + exact_ipca_batched.mean_
    )
    batch_error = torch.norm(
        reconstruction_exact - reconstruction_exact_batched
    ) / torch.norm(X)
    assert batch_error < 1e-4, (
        f"Batched version differs from single batch: {batch_error}"
    )

    # Test that explained variance is in descending order
    assert torch.all(
        exact_ipca.explained_variance_[:-1] >= exact_ipca.explained_variance_[1:]
    ), "Explained variance should be in descending order"

    # Test fit_transform method
    exact_ipca_ft = ExactIncrementalPCA(n_components=n_components)
    X_ft = exact_ipca_ft.fit_transform(X)
    assert X_ft.shape == (n_samples, n_components), (
        "fit_transform output has wrong shape"
    )
