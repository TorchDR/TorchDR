# Author: Mathurin Massias
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch
from sklearn.decomposition import KernelPCA as skKernelPCA

from torchdr.affinity import NormalizedGaussianAffinity, SinkhornAffinity
from torchdr.utils import pykeops

from torchdr import KernelPCA, PHATE

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
