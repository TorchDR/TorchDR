"""
Tests for functions in eval module.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: BSD 3-Clause License

import warnings

import numpy as np
import pytest
import torch
from sklearn.metrics import silhouette_score as sk_silhouette_score
from torch.testing import assert_close

from torchdr.eval import admissible_LIST_METRICS, silhouette_samples, silhouette_score
from torchdr.tests.utils import toy_dataset
from torchdr.utils import pairwise_distances, pykeops

lst_types = ["float32", "float64"]
if pykeops:
    lst_backend = ["keops", None]
else:
    lst_backend = [None]
DEVICE = "cpu"


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "whatever"])
def test_silhouette_score_euclidean(dtype, backend, metric):
    # perfect silhouette
    n = 10
    Id = torch.eye(n, device=DEVICE, dtype=getattr(torch, dtype))
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
            coeffs = silhouette_samples(Id, y_I, None, metric, None, backend, True)
            assert issubclass(w[-1].category, UserWarning)

        assert_close(coeffs, ones)
        weighted_coeffs = silhouette_samples(
            Id, y_I, ones / n, metric, DEVICE, backend, False
        )
        assert_close(coeffs, weighted_coeffs)
        score = silhouette_score(Id, y_I, None, metric, DEVICE, backend)
        assert_close(coeffs.mean(), score)
        sampled_score = silhouette_score(Id, y_I, None, metric, DEVICE, backend, n)
        assert_close(score, sampled_score)

        # tests with equidistant samples
        coeffs_2 = silhouette_samples(Id, y_I2, ones / n, metric, None, backend)
        assert_close(coeffs_2, zeros)

        weighted_coeffs_2 = silhouette_samples(
            Id, y_I2, None, metric, DEVICE, backend, False
        )
        assert_close(coeffs_2, weighted_coeffs_2)

        score_2 = silhouette_score(Id, y_I2, None, metric, DEVICE, backend)
        assert_close(coeffs_2.mean(), score_2)

    else:
        with pytest.raises(ValueError):
            _ = silhouette_samples(Id, y_I, None, metric, None, backend, True)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
def test_silhouette_score_precomputed(dtype, backend):
    # perfect silhouette
    n = 10
    Id = torch.eye(n, device=DEVICE, dtype=getattr(torch, dtype))
    CI, _ = pairwise_distances(Id, Id, "euclidean")
    ones = torch.ones(n, device=DEVICE, dtype=getattr(torch, dtype))

    y_I2 = []
    for i in range(n // 2):
        y_I2 += [i] * 2
    y_I2 = torch.tensor(y_I2, device=DEVICE)

    # tests edge cases with isolated samples
    coeffs_pre = silhouette_samples(CI, y_I2, None, "precomputed", None, backend, True)
    coeffs = silhouette_samples(Id, y_I2, None, "euclidean", None, backend, True)

    assert_close(coeffs_pre, coeffs)
    weighted_coeffs_pre = silhouette_samples(
        CI, y_I2, ones / n, "precomputed", DEVICE, backend, False
    )
    assert_close(coeffs_pre, weighted_coeffs_pre)
    score_pre = silhouette_score(CI, y_I2, None, "precomputed", DEVICE, backend)
    assert_close(coeffs_pre.mean(), score_pre)
    sampled_score_pre = silhouette_score(
        CI, y_I2, None, "precomputed", DEVICE, backend, n
    )
    assert_close(score_pre, sampled_score_pre)

    # catch errors
    with pytest.raises(ValueError):
        _ = silhouette_samples(
            CI[:, :-2], y_I2, None, "precomputed", None, backend, True
        )

    with pytest.raises(ValueError):
        _ = silhouette_score(CI[:, :-2], y_I2, None, "precomputed", None, backend, n)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_consistency_sklearn(dtype, backend, metric):
    n = 100
    X, y = toy_dataset(n, dtype)

    score_torchdr = silhouette_score(X, y, None, metric, DEVICE, backend)
    score_sklearn = sk_silhouette_score(X, y, metric=metric)
    assert (score_torchdr - score_sklearn) ** 2 < 1e-5, (
        "Silhouette scores from torchdr and sklearn should be close."
    )

    # Generate y_noise by permuting half of the indices in y randomly
    y_noise = y.copy()
    y_noise[: n // 2] = np.random.permutation(y_noise[: n // 2])

    score_torchdr_noise = silhouette_score(X, y_noise, None, metric, DEVICE, backend)
    score_sklearn_noise = sk_silhouette_score(X, y_noise, metric=metric)
    assert (score_torchdr_noise - score_sklearn_noise) ** 2 < 1e-5, (
        "Silhouette scores from torchdr and sklearn should be close on noised labels."
    )
