import numpy as np
import torch
from sklearn.decomposition import KernelPCA as skKernelPCA
from sklearn.metrics import pairwise_distances

from torchdr.spectral import KernelPCA
from torchdr.affinity import GibbsAffinity


# def test_fit_transform():
if __name__ == "__main__":
    X = torch.randn(20, 30)
    aff = GibbsAffinity(normalization_dim=None, nodiag=False, sigma=1)
    model = KernelPCA(affinity=aff)

    res_1 = model.fit_transform(X)
    model.fit(X)
    res_2 = model.transform(X)

    np.testing.assert_allclose(res_1, res_2)

    # test = aff.fit_transform(X)

    # dist = pairwise_distances(X)
    # true = torch.exp(- torch.Tensor(dist) ** 2)
