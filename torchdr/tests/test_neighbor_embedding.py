# -*- coding: utf-8 -*-
"""
Tests for neighbor embedding methods.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
import pytest
from sklearn.metrics import silhouette_score

from torchdr.neighbor_embedding import (
    SNE,
    TSNE,
    TSNEkhorn,
    LargeVis,
    InfoTSNE,
    UMAP,
)
from torchdr.tests.utils import toy_dataset
from torchdr.utils import check_shape, pykeops

if pykeops:
    lst_keops = [True, False]
else:
    lst_keops = [False]


lst_types = ["float32", "float64"]
SEA_params = {"lr_affinity_in": 1e-1, "max_iter_affinity_in": 1000}
DEVICE = "cpu"


param_optim = {"lr": 1.0, "optimizer": "Adam", "optimizer_kwargs": None}


@pytest.mark.parametrize(
    "DRModel, kwargs",
    [
        (SNE, {}),
        (TSNE, {}),
        (TSNEkhorn, SEA_params | {"unrolling": True}),
        (TSNEkhorn, SEA_params | {"unrolling": False}),
        (LargeVis, {}),
        (InfoTSNE, {}),
        (UMAP, {}),
    ],
)
@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("keops", lst_keops)
def test_NE(DRModel, kwargs, dtype, keops):
    n = 100
    X, y = toy_dataset(n, dtype)

    model = DRModel(
        n_components=2,
        keops=keops,
        device=DEVICE,
        init="normal",
        max_iter=100,
        random_state=0,
        tol=1e-10,
        **(param_optim | kwargs),
    )
    Z = model.fit_transform(X)

    check_shape(Z, (n, 2))
    assert silhouette_score(Z, y) > 0.2, "Silhouette score should not be too low."


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("keops", lst_keops)
def test_array_init(dtype, keops):
    n = 100
    X, y = toy_dataset(n, dtype)

    Z_init_np = np.random.randn(n, 2).astype(dtype)
    Z_init_torch = torch.from_numpy(Z_init_np)

    torch.use_deterministic_algorithms(True)

    lst_Z = []
    for Z_init in [Z_init_np, Z_init_torch]:
        model = SNE(
            n_components=2,
            keops=keops,
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
    assert (
        (lst_Z[0] - lst_Z[1]) ** 2
    ).mean() < 1e-5, "The two inits should yield similar results."
