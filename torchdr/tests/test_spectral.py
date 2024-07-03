import numpy as np
import torch
from sklearn.decomposition import KernelPCA as skKernelPCA
from sklearn.metrics import pairwise_distances

from torchdr.spectral import KernelPCA
from torchdr.affinity import GibbsAffinity
from torchdr.utils import center_kernel


# def test_fit_transform():
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(4, 5)
    X /= torch.linalg.norm(X, axis=0)  # otherwise all points at distance 1
    aff = GibbsAffinity(normalization_dim=None, nodiag=False, sigma=1)
    model = KernelPCA(affinity=aff, n_components=3)

    res_1 = model.fit_transform(X)
    model.fit(X)
    res_2 = model.transform(X)

    # np.testing.assert_allclose(res_1, res_2)
    K = center_kernel(aff.fit_transform(X))
    print(torch.linalg.eigh(K)[0])
    print(torch.linalg.eigh(K)[1])
    print(model.eigenvectors_)
    # test = aff.fit_transform(X)

    # dist = pairwise_distances(X)
    # true = torch.exp(- torch.Tensor(dist) ** 2)
