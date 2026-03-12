# Author: Mathurin Massias
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch
from sklearn.decomposition import KernelPCA as skKernelPCA
from torch.utils.data import DataLoader, TensorDataset

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
    # Compare only common dimensions since TorchDR removes null space eigenvalues
    n_common = min(X_sk.shape[1], res_1.shape[1])
    np.testing.assert_allclose(
        np.abs(X_sk[:, :n_common]), np.abs(res_1[:, :n_common]), rtol=rtol
    )


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


def test_phate_dataloader_not_supported():
    X, _ = toy_dataset()
    X = torch.tensor(X, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(X), batch_size=32, shuffle=False)
    phate = PHATE(backend=None, device="cpu", max_iter=5)

    with pytest.raises(NotImplementedError, match="does not support DataLoader"):
        phate.fit_transform(dataloader)


def test_phate_landmark_vs_non_landmark_shapes_and_finite():
    X, _ = toy_dataset(n=180)
    X = torch.tensor(X, dtype=torch.float32)

    phate_exact = PHATE(
        backend=None,
        device="cpu",
        max_iter=5,
        random_state=0,
        check_interval=1,
    )
    emb_exact = phate_exact.fit_transform(X)

    phate_landmark = PHATE(
        backend=None,
        device="cpu",
        max_iter=5,
        random_state=0,
        check_interval=1,
        n_landmarks=30,
        random_landmarking=False,
    )
    emb_landmark = phate_landmark.fit_transform(X)

    assert emb_exact.shape == (X.shape[0], 2)
    assert emb_landmark.shape == (X.shape[0], 2)
    assert torch.isfinite(emb_exact).all()
    assert torch.isfinite(emb_landmark).all()


def test_phate_landmark_interpolation_consistency():
    X, _ = toy_dataset(n=200)
    X = torch.tensor(X, dtype=torch.float32)

    phate = PHATE(
        backend=None,
        device="cpu",
        max_iter=5,
        random_state=1,
        check_interval=1,
        n_landmarks=40,
        random_landmarking=False,
    )
    embedding = phate.fit_transform(X)

    assert hasattr(phate, "landmark_embedding_")
    assert torch.isfinite(embedding).all()
    assert torch.isfinite(phate.landmark_embedding_).all()
    assert embedding.shape == (X.shape[0], 2)
    assert phate.landmark_embedding_.shape[0] <= 40
    assert phate.landmark_embedding_.shape[1] == 2


def test_phate_init_scaling_is_respected_in_sgd_init():
    torch.manual_seed(0)
    n = 64
    target_dist = torch.rand(n, n, dtype=torch.float32)
    target_dist = (target_dist + target_dist.T) / 2
    target_dist.fill_diagonal_(0)

    phate = PHATE(
        backend=None,
        device="cpu",
        init="random",
        init_scaling=1e-4,
        max_iter=1,
    )
    embedding = phate._init_embedding_sgd(target_dist)
    first_dim_std = embedding[:, 0].std()

    assert torch.isfinite(embedding).all()
    assert torch.allclose(first_dim_std, torch.tensor(1e-4), rtol=1e-2, atol=1e-8)
