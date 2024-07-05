# Author: Mathurin Massias
#
# License: BSD 3-Clause License

import pytest
import torch
import numpy as np
from sklearn.decomposition import KernelPCA as skKernelPCA

from torchdr.spectral import KernelPCA
from torchdr.affinity import GibbsAffinity, SinkhornAffinity


def test_KernelPCA_sklearn():
    torch.manual_seed(0)
    X = torch.randn(10, 20)
    X /= torch.linalg.norm(X, axis=0)  # otherwise all points at distance 1
    sigma = 2
    n_components = 3
    aff = GibbsAffinity(normalization_dim=None, nodiag=False, sigma=sigma)
    model = KernelPCA(affinity=aff, n_components=n_components)

    # fit then transform does same as fit_transform:
    res_1 = model.fit_transform(X)
    model.fit(X)
    res_2 = model.transform(X)
    np.testing.assert_allclose(res_1, res_2, rtol=1e-6, atol=1e-6)

    # same results as sklearn for Gaussian kernel
    model_sk = skKernelPCA(
        n_components=n_components, kernel="rbf", gamma=1/sigma).fit(X)
    X_sk = model_sk.transform(X)
    np.testing.assert_allclose(X_sk, res_1, rtol=1e-3)


def test_KernelPCA_no_transform():
    torch.manual_seed(0)
    X = torch.randn(10, 20)
    X /= torch.linalg.norm(X, axis=0)
    n_components = 3
    aff = SinkhornAffinity(nodiag=False)
    model = KernelPCA(affinity=aff, n_components=n_components)

    # this should work fine:
    model.fit(X)
    model.fit_transform(X)

    with pytest.raises(
            ValueError, match="Use the fit_transform method instead"):
        model.transform(X)  # cannot use transform.
