import numpy as np

from torchdr.spectral import KernelPCA
from sklearn.decomposition import KernelPCA as skKernelPCA



def test_fit_transform():
    X = np.random.rand(20, 30)
    model = KernelPCA()

    res_1 = model.fit_transform(X)
    model.fit(X)
    res_2 = model.transform(X)

    np.testing.assert_allclose(res_1, res_2)
