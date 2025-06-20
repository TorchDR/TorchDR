# Author: Mathurin Massias
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch
from sklearn.decomposition import KernelPCA as skKernelPCA

from torchdr.affinity import GaussianAffinity, SinkhornAffinity
from torchdr.utils import pykeops

from torchdr import KernelPCA, PHATE

from torchdr.tests.utils import toy_dataset
from torchdr.affinity import NegPotentialAffinity


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
USE_KEOPS = [False]
if pykeops:
    USE_KEOPS.append(True)


@pytest.mark.parametrize("n_components", [3, None])
def test_KernelPCA_sklearn(n_components):
    torch.manual_seed(0)
    X = torch.randn(10, 20)
    X /= torch.linalg.norm(X, axis=0, keepdims=True)
    # otherwise all points at distance 1
    Y = torch.randn(5, 20)
    Y /= torch.linalg.norm(Y, axis=0, keepdims=True)
    sigma = 2
    aff = GaussianAffinity(zero_diag=False, sigma=sigma)
    model = KernelPCA(affinity=aff, n_components=n_components)
    rtol = 1e-2  # we might want to take a look at that someday

    # fit then transform does same as fit_transform:
    res_1 = model.fit_transform(X)
    model.fit(X)
    res_2 = model.transform(X)
    np.testing.assert_allclose(res_1, res_2, rtol=rtol, atol=1e-5)

    # same results as sklearn for Gaussian kernel
    res_Y = model.transform(Y)
    model_sk = skKernelPCA(
        n_components=n_components, kernel="rbf", gamma=1 / sigma
    ).fit(X)
    X_sk = model_sk.transform(X)
    Y_sk = model_sk.transform(Y)
    np.testing.assert_allclose(X_sk, res_1, rtol=rtol)
    np.testing.assert_allclose(Y_sk, res_Y, rtol=rtol)


def test_KernelPCA_no_transform():
    torch.manual_seed(0)
    X = torch.randn(10, 20)
    X /= torch.linalg.norm(X, axis=0)
    n_components = 3
    aff = SinkhornAffinity(zero_diag=False)
    model = KernelPCA(affinity=aff, n_components=n_components)

    # this should work fine:
    model.fit(X)
    model.fit_transform(X)

    match = (
        "can only be used when `affinity` is an UnnormalizedAffinity or "
        "UnnormalizedLogAffinity"
    )
    with pytest.raises(ValueError, match=match):
        model.transform(X)  # cannot use transform.


@pytest.mark.skipif(not pykeops, reason="pykeops is not available")
def test_KernelPCA_keops():
    with pytest.raises(NotImplementedError):
        KernelPCA(backend="keops")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "backend", ["keops", "faiss", None] if pykeops else ["faiss", None]
)
def test_potential_dist(device, backend):
    data, _ = toy_dataset()
    data = torch.tensor(data, dtype=torch.float32)
    data.requires_grad = True
    neg_affinity = NegPotentialAffinity(backend=backend, device=device)(data)
    assert neg_affinity.shape == (data.shape[0], data.shape[0])
    assert neg_affinity.min() < 0
    neg_affinity.sum().backward()
    assert data.grad is not None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "backend", ["keops", "faiss", None] if pykeops else ["faiss", None]
)
def test_phate(device, backend):
    torch.autograd.set_detect_anomaly(True)
    data, _ = toy_dataset()
    data = torch.tensor(data, dtype=torch.float32)
    data = data.to(device)
    phate = PHATE(backend=backend, device=device)
    embedding = phate.fit_transform(data)
    assert embedding.shape == (data.shape[0], 2)
