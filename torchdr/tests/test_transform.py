"""Tests for non-parametric transform on neighbor embedding methods."""

# License: BSD 3-Clause License

import pytest
import torch

from torchdr.neighbor_embedding import UMAP, LargeVis, InfoTSNE, PACMAP, TSNE
from torchdr.tests.utils import toy_dataset
from torchdr.utils import check_shape


DEVICE = "cpu"


@pytest.mark.parametrize(
    "DRModel, kwargs",
    [
        (UMAP, {"n_neighbors": 10, "optimizer": "SGD"}),
        (LargeVis, {"perplexity": 10}),
        (InfoTSNE, {"perplexity": 10, "n_negatives": 10}),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_transform_shape(DRModel, kwargs, dtype):
    """transform(X_new, X_train) returns correct shape."""
    n_train, n_test = 100, 20
    X_train, _ = toy_dataset(n_train, dtype)
    X_test, _ = toy_dataset(n_test, dtype)

    model = DRModel(
        n_components=2,
        device=DEVICE,
        init="normal",
        max_iter=100,
        random_state=0,
        **kwargs,
    )
    model.fit(X_train)
    Z = model.transform(X_test, X_train=X_train)
    check_shape(Z, (n_test, 2))
    assert not torch.isnan(torch.as_tensor(Z)).any(), "Transform produced NaNs."


def test_transform_none_returns_training():
    """transform(None) returns the training embedding."""
    n = 50
    X, _ = toy_dataset(n, "float32")

    model = UMAP(
        n_components=2,
        device=DEVICE,
        init="normal",
        max_iter=50,
        random_state=0,
        n_neighbors=10,
        optimizer="SGD",
    )
    Z_fit = model.fit_transform(X)
    Z_transform = model.transform()
    assert torch.equal(torch.as_tensor(Z_fit), torch.as_tensor(Z_transform))


def test_transform_missing_X_train_raises():
    """transform(X_new) without X_train raises ValueError."""
    n = 50
    X, _ = toy_dataset(n, "float32")

    model = UMAP(
        n_components=2,
        device=DEVICE,
        init="normal",
        max_iter=50,
        random_state=0,
        n_neighbors=10,
        optimizer="SGD",
    )
    model.fit(X)

    X_test, _ = toy_dataset(20, "float32")
    with pytest.raises(ValueError, match="X_train is required"):
        model.transform(X_test)


def test_transform_not_fitted_raises():
    """transform before fit raises ValueError."""
    model = UMAP(n_components=2, n_neighbors=10, optimizer="SGD")
    X_test, _ = toy_dataset(20, "float32")
    with pytest.raises(ValueError, match="not fitted"):
        model.transform(X_test, X_train=X_test)


def test_transform_numpy_input():
    """transform works with numpy input and returns numpy output."""
    n_train, n_test = 80, 15
    X_train, _ = toy_dataset(n_train, "float32")
    X_test, _ = toy_dataset(n_test, "float32")

    model = UMAP(
        n_components=2,
        device=DEVICE,
        init="normal",
        max_iter=50,
        random_state=0,
        n_neighbors=10,
        optimizer="SGD",
    )
    model.fit(X_train)

    # X_train and X_test are numpy arrays from toy_dataset
    Z = model.transform(X_test, X_train=X_train)
    assert Z.shape == (n_test, 2)


def test_embedding_train_stored_on_cpu():
    """embedding_train_ should be stored on CPU after fit."""
    n = 50
    X, _ = toy_dataset(n, "float32")

    model = UMAP(
        n_components=2,
        device=DEVICE,
        init="normal",
        max_iter=50,
        random_state=0,
        n_neighbors=10,
        optimizer="SGD",
    )
    model.fit(X)

    assert hasattr(model, "embedding_train_")
    assert model.embedding_train_.device == torch.device("cpu")
    assert model.embedding_train_.shape == (n, 2)


def test_transform_unsupported_model_raises():
    """Models without bipartite affinity should fail fast in transform."""
    model = PACMAP(n_components=2, n_neighbors=5)
    model.is_fitted_ = True
    model.device_ = DEVICE

    X_test = torch.randn(5, 3)
    X_train = torch.randn(20, 3)

    with pytest.raises(
        NotImplementedError, match="does not support non-parametric transform"
    ):
        model.transform(X_test, X_train=X_train)


def test_transform_auto_lr_reuses_fit_learning_rate():
    """Transform should reuse the fit-time LR when lr='auto'."""
    X, _ = toy_dataset(50, "float32")
    model = LargeVis(
        n_components=2,
        device=DEVICE,
        init="normal",
        max_iter=10,
        random_state=0,
        perplexity=10,
    )
    model.fit(X)

    assert model.lr == "auto"
    expected_fit_lr = max(model.n_samples_in_ / model.early_exaggeration_coeff / 4, 50)
    assert model._get_transform_learning_rate() == pytest.approx(expected_fit_lr / 4.0)
    assert model._get_transform_learning_rate() != pytest.approx(0.25)


def test_transform_negative_sampling_discards_neighbors():
    """Transform negative sampling should exclude nearest neighbors when requested."""
    model = UMAP(
        n_components=2,
        device=DEVICE,
        init="normal",
        max_iter=10,
        random_state=0,
        n_neighbors=2,
        negative_sample_rate=1,
        discard_NNs=True,
        optimizer="SGD",
    )
    model.device_ = DEVICE

    nn_indices = torch.tensor([[0, 1], [1, 3]])
    neg_indices = model._sample_transform_neg_indices(
        n_new=2, n_train=5, nn_indices=nn_indices
    )
    neg_local = neg_indices - 2

    assert neg_local.min() >= 0
    assert neg_local.max() < 5
    assert not (neg_local.unsqueeze(-1) == nn_indices.unsqueeze(1)).any()


def test_embedding_train_not_stored_for_non_transform_model():
    """Models without non-parametric transform should not keep a CPU clone."""
    X, _ = toy_dataset(40, "float32")
    model = TSNE(
        n_components=2,
        device=DEVICE,
        init="normal",
        max_iter=5,
        random_state=0,
        perplexity=10,
    )
    model.fit(X)

    assert not hasattr(model, "embedding_train_")
