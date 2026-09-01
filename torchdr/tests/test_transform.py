"""Tests for non-parametric transform on neighbor embedding methods."""

# License: BSD 3-Clause License

import numpy as np
import pytest
import torch

from torchdr.neighbor_embedding import UMAP, LargeVis, InfoTSNE, PACMAP, TSNE
from torchdr.tests.utils import toy_dataset
from torchdr.utils import check_shape


DEVICE = "cpu"
BACKEND = None


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
        backend=BACKEND,
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
        backend=BACKEND,
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
        backend=BACKEND,
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
    model = UMAP(n_components=2, n_neighbors=10, optimizer="SGD", backend=BACKEND)
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
        backend=BACKEND,
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
    assert isinstance(Z, np.ndarray)
    assert Z.shape == (n_test, 2)


def test_embedding_train_stored_on_cpu():
    """embedding_train_ should be stored on CPU after fit."""
    n = 50
    X, _ = toy_dataset(n, "float32")

    model = UMAP(
        n_components=2,
        backend=BACKEND,
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
    model = PACMAP(n_components=2, n_neighbors=5, backend=BACKEND)
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
        backend=BACKEND,
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
        backend=BACKEND,
        device=DEVICE,
        init="normal",
        max_iter=10,
        random_state=0,
        n_neighbors=2,
        negative_sample_rate=1,
        exclude_neighbors_from_negative_sampling=True,
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


def test_negative_sampling_handles_padding_and_duplicate_exclusions():
    """The shared sampler must ignore invalid and repeated exclusion entries."""
    exclusion = torch.tensor([[-1, 1, 1, 4], [0, 2, 2, 99]])
    adjusted, n_available = UMAP._prepare_exclusion_sampling(exclusion, n_candidates=5)
    draws = UMAP._draw_with_exclusions(adjusted, n_available, n_draws=2000)

    assert draws.min() >= 0
    assert draws.max() < 5
    assert not torch.isin(draws[0], torch.tensor([1, 4])).any()
    assert not torch.isin(draws[1], torch.tensor([0, 2])).any()
    assert torch.equal(torch.unique(draws[0]).sort().values, torch.tensor([0, 2, 3]))
    assert torch.equal(torch.unique(draws[1]).sort().values, torch.tensor([1, 3, 4]))


def test_negative_sampling_rejects_fully_excluded_rows():
    """A row with no valid negative candidate should fail deterministically."""
    exclusion = torch.tensor([[0, 1, 1, -1]])
    with pytest.raises(ValueError, match="No candidates remain"):
        UMAP._prepare_exclusion_sampling(exclusion, n_candidates=2)


def test_umap_transform_init_uses_exact_neighbor_embedding():
    """UMAP should snap only zero-distance matches, not arbitrary strong edges."""
    model = UMAP(
        n_components=2,
        backend=BACKEND,
        device=DEVICE,
        init="normal",
        max_iter=30,
        random_state=0,
        n_neighbors=2,
        optimizer="SGD",
    )
    train_emb = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    affinity = torch.tensor([[1.0, 0.25], [0.25, 0.75]])
    nn_indices = torch.tensor([[1, 2], [0, 2]])
    neighbor_distances = torch.tensor([[0.0, 1.0], [0.1, 0.2]])

    embedding_new = model._initialize_transform_embedding(
        affinity, nn_indices, train_emb, neighbor_distances=neighbor_distances
    )

    assert torch.equal(embedding_new[0], train_emb[1])
    expected = 0.25 * train_emb[0] + 0.75 * train_emb[2]
    assert torch.allclose(embedding_new[1], expected)


def test_umap_bipartite_affinity_only_marks_exact_matches():
    """A nonzero nearest distance must not be assigned unit UMAP affinity."""
    model = UMAP(
        n_neighbors=3,
        backend=BACKEND,
        device=DEVICE,
        optimizer="SGD",
        max_iter_affinity=100,
    )
    distances = torch.tensor([[0.2, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=torch.float64)
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]])

    affinity = model._compute_bipartite_affinity(distances, indices)

    assert affinity[0, 0] < 1
    assert affinity[1, 0] == 1
    target_marginal = torch.log2(torch.tensor(3.0, dtype=affinity.dtype))
    assert torch.allclose(
        affinity[:, 1:].sum(dim=1),
        target_marginal.expand(2),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("DRModel", [LargeVis, InfoTSNE])
def test_entropic_transform_affinity_matches_fit_invariants(DRModel):
    """Transform affinities should preserve perplexity and total unit mass."""
    model = DRModel(
        perplexity=3,
        backend=BACKEND,
        device=DEVICE,
        max_iter_affinity=200,
    )
    distances = torch.tensor(
        [
            [0.0, 0.2, 0.5, 1.0, 1.8, 2.9, 4.3, 6.0, 8.0],
            [0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9, 6.4, 8.1],
        ],
        dtype=torch.float64,
    )
    indices = torch.arange(9).expand(2, -1)

    affinity = model._compute_bipartite_affinity(distances, indices)
    row_probabilities = affinity / affinity.sum(dim=1, keepdim=True)
    shannon_entropy = -(row_probabilities * row_probabilities.log()).sum(dim=1)

    assert model._get_n_neighbors_transform(n_train=100) == 9
    assert torch.allclose(
        affinity.sum(dim=1), torch.full((2,), 0.5, dtype=affinity.dtype)
    )
    assert affinity.sum().item() == pytest.approx(1.0)
    assert torch.allclose(
        shannon_entropy,
        torch.log(torch.tensor(3.0, dtype=affinity.dtype)).expand(2),
        atol=1e-5,
        rtol=1e-5,
    )
    assert not torch.allclose(
        row_probabilities,
        torch.full_like(row_probabilities, 1 / row_probabilities.shape[1]),
    )

    boundary_affinity = model._compute_bipartite_affinity(
        distances[:, :3], indices[:, :3]
    )
    assert torch.equal(boundary_affinity, torch.full_like(boundary_affinity, 1 / 6))


def test_umap_transform_uses_epoch_schedule():
    """UMAP transform should keep fit-style epoch sampling instead of all-edges updates."""
    model = UMAP(
        n_components=2,
        backend=BACKEND,
        device=DEVICE,
        init="normal",
        max_iter=90,
        random_state=0,
        n_neighbors=2,
        optimizer="SGD",
    )
    model.device_ = torch.device(DEVICE)

    embedding_new = torch.zeros(2, 2)
    train_emb = torch.zeros(4, 2)
    affinity = torch.tensor([[1.0, 0.1], [0.05, 0.01]])
    nn_indices = torch.tensor([[0, 1], [2, 3]])

    saved = model._enter_transform(embedding_new, train_emb, affinity, nn_indices)
    try:
        assert model.n_samples_in_ == embedding_new.shape[0]
        assert torch.equal(model.epoch_of_next_sample, model.epochs_per_sample)
        assert not torch.equal(
            model.epochs_per_sample, torch.zeros_like(model.epochs_per_sample)
        )
        assert torch.isfinite(model.epochs_per_sample[0, 0])
        assert torch.isinf(model.epochs_per_sample[1, 1])
    finally:
        model._exit_transform(saved)

    assert model.embedding_ is None
    assert not hasattr(model, "n_samples_in_")
    assert not hasattr(model, "epochs_per_sample")


def test_transform_restores_registered_parameter_embedding():
    """Temporary concatenation must not corrupt Parameter registration."""
    model = UMAP(
        n_neighbors=2,
        backend=BACKEND,
        device=DEVICE,
        optimizer="SGD",
        max_iter=6,
    )
    model.device_ = torch.device(DEVICE)
    original_embedding = torch.nn.Parameter(torch.randn(4, 2))
    model.embedding_ = original_embedding
    embedding_new = torch.zeros(2, 2)
    train_emb = original_embedding.detach()
    affinity = torch.tensor([[1.0, 0.2], [0.8, 0.4]])
    nn_indices = torch.tensor([[0, 1], [2, 3]])

    saved = model._enter_transform(embedding_new, train_emb, affinity, nn_indices)
    model.embedding_ = torch.cat([embedding_new, train_emb], dim=0)
    model._exit_transform(saved)

    assert model.embedding_ is original_embedding
    assert model._parameters["embedding_"] is original_embedding


def test_transform_validates_reference_shape_and_features():
    """Transform should reject references that cannot align with the fitted model."""
    model = UMAP(n_neighbors=2, backend=BACKEND, device=DEVICE, optimizer="SGD")
    model.is_fitted_ = True
    model.device_ = torch.device(DEVICE)
    model.n_features_in_ = 3
    model.embedding_train_ = torch.randn(5, 2)

    with pytest.raises(ValueError, match="X_new has 4 features"):
        model.transform(torch.randn(2, 4), X_train=torch.randn(5, 3))
    with pytest.raises(ValueError, match="X_train has 4 features"):
        model.transform(torch.randn(2, 3), X_train=torch.randn(5, 4))
    with pytest.raises(ValueError, match="same training samples"):
        model.transform(torch.randn(2, 3), X_train=torch.randn(4, 3))


def test_duplicate_training_rows_keep_reference_embedding_aligned():
    """The stored reference must use the original, not deduplicated, row space."""
    X_unique = torch.randn(8, 3)
    X_train = torch.cat([X_unique, X_unique[:2]], dim=0)
    model = UMAP(
        n_neighbors=2,
        backend=BACKEND,
        device=DEVICE,
        init="normal",
        max_iter=6,
        random_state=0,
        optimizer="SGD",
    )

    model.fit(X_train)
    fit_embedding = model.embedding_
    fit_sample_count = model.n_samples_in_
    transformed = model.transform(torch.randn(2, 3), X_train=X_train)

    assert model.embedding_train_.shape[0] == X_train.shape[0]
    assert transformed.shape == (2, 2)
    assert model.embedding_ is fit_embedding
    assert model.n_samples_in_ == fit_sample_count
    assert not hasattr(model, "neg_indices_")


def test_negative_sampling_parameter_alias_is_backward_compatible():
    """The pre-PR public parameter remains usable during its deprecation window."""
    with pytest.warns(FutureWarning, match="discard_NNs"):
        model = UMAP(discard_NNs=True)
    assert model.exclude_neighbors_from_negative_sampling is True

    with pytest.raises(ValueError, match="Conflicting values"):
        UMAP(
            exclude_neighbors_from_negative_sampling=True,
            discard_NNs=False,
        )

    assert PACMAP().exclude_neighbors_from_negative_sampling is True


def test_embedding_train_not_stored_for_non_transform_model():
    """Models without non-parametric transform should not keep a CPU clone."""
    X, _ = toy_dataset(40, "float32")
    model = TSNE(
        n_components=2,
        backend=BACKEND,
        device=DEVICE,
        init="normal",
        max_iter=5,
        random_state=0,
        perplexity=10,
    )
    model.fit(X)

    assert not hasattr(model, "embedding_train_")
