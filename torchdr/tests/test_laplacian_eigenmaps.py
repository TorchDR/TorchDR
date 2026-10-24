"""Independent graph and estimator checks for Laplacian Eigenmaps."""

import numpy as np
import pytest
import torch
from scipy.sparse.csgraph import laplacian
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.manifold import spectral_embedding

from torchdr import Affinity, LaplacianEigenmaps, NormalizedGaussianAffinity


class FixedAffinity(Affinity):
    """Return an explicit graph for reference and invalid-input checks."""

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def _compute_affinity(self, X):
        return self.graph


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_components", [1, 2, 4])
def test_laplacian_reference(dtype, n_components):
    """Match SciPy's generalized eigenproblem and sklearn's embedding."""
    X = torch.randn(35, 4, generator=torch.Generator().manual_seed(12), dtype=dtype)
    affinity = NormalizedGaussianAffinity(sigma=5, normalization_dim=None)
    W = affinity(X).numpy()
    model = LaplacianEigenmaps(affinity=affinity, n_components=n_components)
    actual = model.fit_transform(X)
    reference = spectral_embedding(
        W, n_components=n_components, eigen_solver="arpack", random_state=0
    )
    # Signs (and bases in a repeated eigenspace) do not define the embedding.
    np.testing.assert_allclose(
        actual.numpy() @ actual.numpy().T,
        reference @ reference.T,
        rtol=2e-4,
        atol=2e-5,
    )
    L = laplacian(W, normed=False)
    D = W.sum(1)
    np.testing.assert_allclose(
        L @ actual.numpy(),
        D[:, None] * actual.numpy() * model.eigenvalues_.numpy(),
        rtol=2e-4,
        atol=2e-5,
    )
    assert actual.dtype == dtype
    assert actual.device == X.device
    assert model.is_fitted_


def test_laplacian_iris_numpy():
    """Support a real dataset and preserve the NumPy input/output contract."""
    X = load_iris().data
    model = LaplacianEigenmaps(process_duplicates=False)
    actual = model.fit_transform(X)
    reference = spectral_embedding(
        NormalizedGaussianAffinity(normalization_dim=None)(torch.from_numpy(X)).numpy(),
        n_components=2,
        random_state=0,
    )
    np.testing.assert_allclose(actual @ actual.T, reference @ reference.T, atol=1e-7)
    assert isinstance(actual, np.ndarray)
    assert clone(model).affinity is None
    assert clone(model).process_duplicates is False
    assert model.fit(X) is model


def test_laplacian_duplicates():
    """Base-class duplicate expansion restores identical sample embeddings."""
    X = torch.randn(12, 3, generator=torch.Generator().manual_seed(2))
    actual = LaplacianEigenmaps().fit_transform(torch.cat([X, X[:2]]))
    torch.testing.assert_close(actual[:2], actual[-2:])


@pytest.mark.parametrize("n_components", [0, -1, 4, 1.5, True])
def test_laplacian_invalid_components(n_components):
    """Reject dimensions that cannot select nontrivial eigenvectors."""
    with pytest.raises(ValueError, match="n_components"):
        LaplacianEigenmaps(n_components=n_components).fit_transform(torch.randn(4, 2))


def test_laplacian_disconnected_graph():
    """Warn about multiple null modes and keep isolated vertices finite."""
    W = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.warns(UserWarning, match="not fully connected"):
        result = LaplacianEigenmaps(
            affinity=FixedAffinity(W),
            n_components=1,
        ).fit_transform(torch.arange(3.0).reshape(-1, 1))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("value", [-1.0, float("nan")])
def test_laplacian_invalid_graph(value):
    """Reject graphs that do not define a nonnegative finite Laplacian."""
    W = torch.ones(4, 4)
    W[0, 1] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        LaplacianEigenmaps(affinity=FixedAffinity(W)).fit_transform(torch.randn(4, 2))


def test_laplacian_dense_backend():
    """Reject lazy backends before starting an unsupported eigensolve."""
    with pytest.raises(NotImplementedError, match="dense backend"):
        LaplacianEigenmaps(backend="keops")
