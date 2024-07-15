# Author: Mathurin Massias
#
# License: BSD 3-Clause License

import pytest
import torch
import numpy as np
from sklearn.decomposition import KernelPCA as skKernelPCA

from torchdr.utils import pykeops
from torchdr.spectral import KernelPCA
from torchdr.affinity import GaussianAffinity, SinkhornAffinity


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
        KernelPCA(keops=True)


if __name__ == "__main__":
    pass
