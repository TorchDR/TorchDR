# -*- coding: utf-8 -*-
"""
Tests for neighbor embedding methods.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score

from torchdr.neighbor_embedding import SNE, TSNE, SNEkhorn, TSNEkhorn
from torchdr.utils import check_shape


lst_types = ["float32", "float64"]
SEA_params = {"lr_affinity_in": 1e-1, "max_iter_affinity_in": 100}

DEVICE = "cpu"


def toy_dataset(n=300, dtype="float32"):
    X, y = make_moons(n_samples=n, noise=0.05, random_state=0)
    return X.astype(dtype), y


@pytest.mark.parametrize(
    "DRModel, kwargs",
    [
        (SNE, {}),
        (TSNE, {}),
        (SNEkhorn, SEA_params | {"unrolling": True}),
        (SNEkhorn, SEA_params | {"unrolling": False}),
        (TSNEkhorn, SEA_params | {"unrolling": True}),
        (TSNEkhorn, SEA_params | {"unrolling": False}),
    ],
)
@pytest.mark.parametrize("dtype", lst_types)
def test_SNE(DRModel, kwargs, dtype):
    n = 300
    X, y = toy_dataset(n, dtype)

    for keops in [False, True]:
        model = DRModel(
            n_components=2,
            perplexity=30,
            keops=keops,
            device=DEVICE,
            init="normal",
            max_iter=100,
            **kwargs
        )
        Z = model.fit_transform(X)

        check_shape(Z, (n, 2))
        assert silhouette_score(Z, y) > 0.2, "Silhouette score should not be too low."
