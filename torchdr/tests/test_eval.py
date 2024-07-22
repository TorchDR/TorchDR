# -*- coding: utf-8 -*-
"""
Tests for functions in eval module.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
import pytest
import warnings
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.manifold import trustworthiness as sk_trustworthiness

from sklearn.decomposition import PCA

from torch.testing import assert_close

from torchdr.eval import (
    admissible_LIST_METRICS,
    silhouette_samples,
    silhouette_score,
    trustworthiness,
    Kary_preservation_score,
)

from torchdr.utils import pykeops, pairwise_distances
from torchdr.tests.utils import toy_dataset

lst_types = ["float32", "float64"]
if pykeops:
    lst_keops = [True, False]
else:
    lst_keops = [False]
DEVICE = "cpu"


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("keops", lst_keops)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "whatever"])
def test_silhouette_score_euclidean(dtype, keops, metric):
    # perfect silhouette
    n = 10
    I = torch.eye(n, device=DEVICE, dtype=getattr(torch, dtype))
    y_I = torch.arange(n, device=DEVICE)
    ones = torch.ones(n, device=DEVICE, dtype=getattr(torch, dtype))
    zeros = torch.zeros(n, device=DEVICE, dtype=getattr(torch, dtype))

    y_I2 = []
    for i in range(n // 2):
        y_I2 += [i] * 2
    y_I2 = torch.tensor(y_I2, device=DEVICE)

    if metric in admissible_LIST_METRICS:
        # tests edge cases with isolated samples
        with warnings.catch_warnings(record=True) as w:
            coeffs = silhouette_samples(I, y_I, None, metric, None, keops, True)
            assert issubclass(w[-1].category, UserWarning)

        assert_close(coeffs, ones)
        weighted_coeffs = silhouette_samples(
            I, y_I, ones / n, metric, DEVICE, keops, False
        )
        assert_close(coeffs, weighted_coeffs)
        score = silhouette_score(I, y_I, None, metric, DEVICE, keops)
        assert_close(coeffs.mean(), score)
        sampled_score = silhouette_score(I, y_I, None, metric, DEVICE, keops, n)
        assert_close(score, sampled_score)

        # tests with equidistant samples
        coeffs_2 = silhouette_samples(I, y_I2, ones / n, metric, None, keops)
        assert_close(coeffs_2, zeros)

        weighted_coeffs_2 = silhouette_samples(
            I, y_I2, None, metric, DEVICE, keops, False
        )
        assert_close(coeffs_2, weighted_coeffs_2)

        score_2 = silhouette_score(I, y_I2, None, metric, DEVICE, keops)
        assert_close(coeffs_2.mean(), score_2)

    else:
        with pytest.raises(ValueError):
            _ = silhouette_samples(I, y_I, None, metric, None, keops, True)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("keops", lst_keops)
def test_silhouette_score_precomputed(dtype, keops):
    # perfect silhouette
    n = 10
    I = torch.eye(n, device=DEVICE, dtype=getattr(torch, dtype))
    CI = pairwise_distances(I, I, "euclidean")
    ones = torch.ones(n, device=DEVICE, dtype=getattr(torch, dtype))

    y_I2 = []
    for i in range(n // 2):
        y_I2 += [i] * 2
    y_I2 = torch.tensor(y_I2, device=DEVICE)

    # tests edge cases with isolated samples
    coeffs_pre = silhouette_samples(CI, y_I2, None, "precomputed", None, keops, True)
    coeffs = silhouette_samples(I, y_I2, None, "euclidean", None, keops, True)

    assert_close(coeffs_pre, coeffs)
    weighted_coeffs_pre = silhouette_samples(
        CI, y_I2, ones / n, "precomputed", DEVICE, keops, False
    )
    assert_close(coeffs_pre, weighted_coeffs_pre)
    score_pre = silhouette_score(CI, y_I2, None, "precomputed", DEVICE, keops)
    assert_close(coeffs_pre.mean(), score_pre)
    sampled_score_pre = silhouette_score(
        CI, y_I2, None, "precomputed", DEVICE, keops, n
    )
    assert_close(score_pre, sampled_score_pre)

    # catch errors
    with pytest.raises(ValueError):
        _ = silhouette_samples(CI[:, :-2], y_I2, None, "precomputed", None, keops, True)

    with pytest.raises(ValueError):
        _ = silhouette_score(CI[:, :-2], y_I2, None, "precomputed", None, keops, n)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("keops", lst_keops)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_silhouette_consistency_sklearn(dtype, keops, metric):
    n = 50
    X, y = toy_dataset(n, dtype)

    score_torchdr = silhouette_score(X, y, None, metric, DEVICE, keops)
    score_sklearn = sk_silhouette_score(X, y, metric=metric)
    assert (
        score_torchdr - score_sklearn
    ) ** 2 < 1e-5, "Silhouette scores from torchdr and sklearn should be close."

    # Generate y_noise by permuting half of the indices in y randomly
    y_noise = y.copy()
    y_noise[: n // 2] = np.random.permutation(y_noise[: n // 2])

    score_torchdr_noise = silhouette_score(X, y_noise, None, metric, DEVICE, keops)
    score_sklearn_noise = sk_silhouette_score(X, y_noise, metric=metric)
    assert (
        score_torchdr_noise - score_sklearn_noise
    ) ** 2 < 1e-5, (
        "Silhouette scores from torchdr and sklearn should be close on noised labels."
    )


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "whatever"])
def test_trustworthiness_euclidean(dtype, metric):
    # perfect trustworthiness
    n = 30
    X = torch.eye(n, device=DEVICE, dtype=getattr(torch, dtype))
    Z = X.clone()
    n_neighbors = 5

    if metric in admissible_LIST_METRICS:
        CX = pairwise_distances(X, X, metric, keops=False)
        CZ = pairwise_distances(Z, Z, metric, keops=False)

        score = trustworthiness(X, Z, n_neighbors, metric, DEVICE)
        # compare to precomputed scores
        with pytest.raises(ValueError):
            _ = trustworthiness(CX[:, :-2], CZ, n_neighbors, "precomputed", DEVICE)

        # compare to precomputed scores
        with pytest.raises(ValueError):
            _ = trustworthiness(CZ, CX[:, :-2], n_neighbors, "precomputed", DEVICE)

        score_precomputed = trustworthiness(CX, CZ, n_neighbors, "precomputed")

        assert score == 1.0
        assert score == score_precomputed

    else:
        with pytest.raises(ValueError):
            _ = trustworthiness(X, Z, n_neighbors, metric, DEVICE)

    # catch errors
    n_neighbors = n / 2 + 1

    with pytest.raises(ValueError):
        _ = trustworthiness(X, Z, n_neighbors, metric, DEVICE)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_trustworthiness_consistency_sklearn(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)
    Z = PCA(n_components=1, random_state=0).fit_transform(X)

    score_torchdr = trustworthiness(X, Z, n_neighbors=5, metric=metric, device=DEVICE)
    score_sklearn = sk_trustworthiness(X, Z, n_neighbors=5, metric=metric)
    assert (
        score_torchdr - score_sklearn
    ) ** 2 < 1e-5, "Trustworthiness from torchdr and sklearn should be close."
