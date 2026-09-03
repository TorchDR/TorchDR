"""
Tests for neighbor embedding methods.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch
from sklearn.metrics import silhouette_score

import torchdr.neighbor_embedding.pacmap as pacmap_module
from torchdr.neighbor_embedding import (
    SNE,
    TSNE,
    COSNE,
    UMAP,
    InfoTSNE,
    LargeVis,
    TSNEkhorn,
    PACMAP,
)
from torchdr.tests.utils import toy_dataset, iris_dataset
from torchdr.utils import check_shape, pykeops

if pykeops:
    lst_backend = ["keops", None]
else:
    lst_backend = [None]


lst_types = ["float32", "float64"]
SEA_params = {"lr_affinity_in": 1e-1, "max_iter_affinity_in": 1000}
DEVICE = "cpu"


param_optim = {"lr": 1.0, "optimizer": "Adam", "optimizer_kwargs": None}


def test_umap_default_learning_rate():
    """UMAP's documented default should configure SGD with an initial LR of 1."""
    model = UMAP(backend=None)
    model.embedding_ = torch.zeros(4, 2, requires_grad=True)
    model._set_params()
    model._set_learning_rate()
    model._configure_optimizer()

    assert model.lr == 1.0
    assert model.lr_ == 1.0
    assert model.optimizer_.param_groups[0]["lr"] == pytest.approx(1.0)


def test_pacmap_mid_near_indices_are_global(monkeypatch):
    """PACMAP should map selected candidate positions to sample indices."""
    n_samples = 10
    model = PACMAP(
        n_neighbors=2,
        MN_ratio=0.5,
        backend=None,
        device=DEVICE,
    )
    model.n_samples_in_ = n_samples
    model.X_ = torch.arange(n_samples, dtype=torch.float32).unsqueeze(1)
    model.embedding_ = torch.zeros(n_samples, 2)
    model.NN_indices_ = torch.zeros(n_samples, 2, dtype=torch.long)
    model.self_idxs = torch.arange(n_samples).unsqueeze(1)
    model.mid_near_indices = torch.empty(n_samples, 1, dtype=torch.long)
    model.w_NB = 1
    model.w_MN = 1

    compressed_candidates = torch.tensor([1, 6, 2, 3, 4, 5]).repeat(n_samples, 1)
    observed = {}

    def fake_randint(low, high, size, device=None):
        assert (low, high, size) == (1, n_samples - 1, (n_samples, 6))
        return compressed_candidates.clone().to(device)

    call_count = 0

    def fake_pairwise_distances_indexed(*args, key_indices, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return torch.zeros(n_samples, model.n_neighbors)
        if call_count == 2:
            observed["candidates"] = key_indices.clone()
            return torch.arange(6, dtype=torch.float32).repeat(n_samples, 1)
        observed["selected"] = key_indices.clone()
        return torch.zeros(n_samples, model.n_mid_near)

    monkeypatch.setattr(pacmap_module.torch, "randint", fake_randint)
    monkeypatch.setattr(
        pacmap_module,
        "pairwise_distances_indexed",
        fake_pairwise_distances_indexed,
    )

    model._compute_attractive_loss()

    expected = observed["candidates"][:, 1]
    assert torch.equal(observed["selected"].squeeze(1), expected)
    assert torch.all(expected > 5)


@pytest.mark.parametrize(
    "DRModel, kwargs",
    [
        (SNE, {}),
        (TSNE, {}),
        (TSNEkhorn, {**SEA_params, "unrolling": True}),
        (TSNEkhorn, {**SEA_params, "unrolling": False}),
        (LargeVis, {}),
        (InfoTSNE, {}),
        (UMAP, {"optimizer": "SGD"}),
        (PACMAP, {}),
    ],
)
@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
def test_NE(DRModel, kwargs, dtype, backend):
    n = 100
    X, y = toy_dataset(n, dtype)

    model = DRModel(
        n_components=2,
        backend=backend,
        device=DEVICE,
        init="normal",
        max_iter=100,
        random_state=0,
        min_grad_norm=1e-10,
        **{**param_optim, **kwargs},
    )
    Z = model.fit_transform(X)

    check_shape(Z, (n, 2))
    assert silhouette_score(Z, y) > 0.15, "Silhouette score should not be too low."


@pytest.mark.parametrize("dtype", lst_types)
def test_COSNE(dtype):
    X, y = iris_dataset(dtype)

    model = COSNE(
        lr=5e-2,
        n_components=2,
        device=DEVICE,
        max_iter=2000,
        random_state=0,
        gamma=1,
        learning_rate_for_h_loss=0.01,
        init_scaling=0.01,
    )
    Z = model.fit_transform(X)

    check_shape(Z, (X.shape[0], 2))
    assert not np.isnan(Z).any(), "COSNE embedding has NaNs."
    assert silhouette_score(Z, y) > 0.15, "Silhouette score should not be too low."


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
def test_array_init(dtype, backend):
    n = 100
    X, y = toy_dataset(n, dtype)

    Z_init_np = np.random.randn(n, 2).astype(dtype)
    Z_init_torch = torch.from_numpy(Z_init_np)

    torch.use_deterministic_algorithms(True)

    lst_Z = []
    for Z_init in [Z_init_np, Z_init_torch]:
        model = SNE(
            n_components=2,
            backend=backend,
            device=DEVICE,
            init=Z_init,
            max_iter=100,
            random_state=0,
            **param_optim,
        )
        Z = model.fit_transform(X)
        lst_Z.append(Z)

        check_shape(Z, (n, 2))
        assert silhouette_score(Z, y) > 0.2, "Silhouette score should not be too low."

    # --- checks that the two inits yield similar results ---
    assert ((lst_Z[0] - lst_Z[1]) ** 2).mean() < 1e-5, (
        "The two inits should yield similar results."
    )
