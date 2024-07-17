# -*- coding: utf-8 -*-
"""
Tests for functions in eval module.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: BSD 3-Clause License

import torch
import pytest
import warnings
from sklearn.metrics import silhouette_score as sk_silhouette_score

from torch.testing import assert_close

from torchdr.eval import silhouette_samples, silhouette_score, admissible_LIST_METRICS
from torchdr.utils import pykeops
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
            coeffs = silhouette_samples(I, y_I, None, metric, None, keops, True)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("keops", lst_keops)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_consistency_sklearn(dtype, keops, metric):
    n = 100
    X, y = toy_dataset(n, dtype)
    print(X.shape)
    score_torchdr = silhouette_score(X, y, None, metric, DEVICE, keops)
    score_sklearn = sk_silhouette_score(X, y, metric=metric)
    assert (
        score_torchdr - score_sklearn
    ) ** 2 < 1e-5, "Silhouette scores from torchdr and sklearn should be close."
