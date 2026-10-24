"""Independent references for standard Locally Linear Embedding."""

import numpy as np
import pytest
import torch
from scipy.linalg import solve
from sklearn.base import clone
from sklearn.datasets import make_s_curve
from sklearn.manifold import locally_linear_embedding
from sklearn.neighbors import NearestNeighbors

from torchdr import LocallyLinearEmbedding


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_neighbors", [6, 10])
@pytest.mark.parametrize("n_components", [1, 2])
def test_lle_reference(dtype, n_neighbors, n_components):
    """Compare independent local solves, embedding subspaces and energy."""
    X = torch.randn(40, 5, dtype=dtype, generator=torch.Generator().manual_seed(4))
    model = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=n_components, process_duplicates=False
    )
    Y = model.fit_transform(X)
    data = X.double().numpy()
    indices = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(data)
        .kneighbors(return_distance=False)
    )
    weights = []
    for i, neighbors in enumerate(indices):
        delta = data[neighbors] - data[i]
        covariance = delta @ delta.T
        trace = np.trace(covariance)
        covariance += model.reg * (trace if trace > 0 else 1) * np.eye(n_neighbors)
        row = solve(covariance, np.ones(n_neighbors), assume_a="pos")
        weights.append(row / row.sum())
    np.testing.assert_array_equal(model.neighbors_.numpy(), indices)
    np.testing.assert_allclose(model.weights_.numpy(), weights, rtol=2e-3, atol=2e-4)
    reference, error = locally_linear_embedding(
        data,
        n_neighbors=n_neighbors,
        n_components=n_components,
        eigen_solver="dense",
        reg=model.reg,
        method="standard",
    )
    np.testing.assert_allclose(
        Y.numpy() @ Y.numpy().T, reference @ reference.T, rtol=2e-3, atol=2e-4
    )
    W = np.zeros((len(data), len(data)))
    W[np.arange(len(data))[:, None], indices] = weights
    energy = np.linalg.norm(Y.numpy() - W @ Y.numpy()) ** 2
    np.testing.assert_allclose(
        model.reconstruction_error_, energy, atol=1e-5, rtol=2e-3
    )
    np.testing.assert_allclose(model.reconstruction_error_, error, atol=1e-5, rtol=2e-3)
    torch.testing.assert_close(
        Y.mean(0), torch.zeros(n_components, dtype=dtype), atol=1e-6, rtol=0
    )
    assert Y.dtype == dtype
    assert Y.device == X.device
    assert model.is_fitted_


def test_lle_s_curve_numpy():
    """Recover sklearn's standard embedding of a curved 3D manifold."""
    X, _ = make_s_curve(120, random_state=3)
    model = LocallyLinearEmbedding(n_neighbors=10, process_duplicates=False)
    Y = model.fit_transform(X)
    reference, _ = locally_linear_embedding(
        X, n_neighbors=10, n_components=2, eigen_solver="dense"
    )
    np.testing.assert_allclose(Y @ Y.T, reference @ reference.T, atol=1e-7)
    assert isinstance(Y, np.ndarray)
    assert clone(model).process_duplicates is False
    assert model.fit(X) is model


def test_lle_duplicates():
    """Expand duplicate rows and keep neighbor metadata in unique-sample space."""
    X = torch.randn(15, 4, generator=torch.Generator().manual_seed(3))
    model = LocallyLinearEmbedding()
    Y = model.fit_transform(torch.cat([X, X[:2]]))
    torch.testing.assert_close(Y[:2], Y[-2:])
    assert model.neighbors_.shape == (15, 5)


def test_lle_zero_trace():
    """Use unscaled regularization for coincident local neighborhoods."""
    model = LocallyLinearEmbedding(n_neighbors=3, process_duplicates=False)
    Y = model.fit_transform(torch.zeros(5, 3, dtype=torch.float64))
    torch.testing.assert_close(model.weights_, torch.full((5, 3), 1 / 3, dtype=Y.dtype))
    assert torch.isfinite(Y).all()


@pytest.mark.parametrize("name", ["n_neighbors", "n_components"])
@pytest.mark.parametrize("value", [0, 5, 1.5])
def test_lle_invalid_dimensions(name, value):
    """Reject dimensions incompatible with the fitted sample count."""
    with pytest.raises(ValueError, match=name):
        params = {"n_neighbors": 3, name: value}
        LocallyLinearEmbedding(**params).fit_transform(torch.randn(5, 3))


@pytest.mark.parametrize("reg", [-1, float("nan"), float("inf")])
def test_lle_invalid_regularization(reg):
    """Require finite nonnegative local regularization."""
    with pytest.raises(ValueError, match="reg"):
        LocallyLinearEmbedding(n_neighbors=3, reg=reg).fit_transform(torch.randn(8, 4))


def test_lle_dense_backend():
    """Fail early for unsupported lazy neighbor-search backends."""
    with pytest.raises(NotImplementedError, match="dense backend"):
        LocallyLinearEmbedding(backend="keops")
