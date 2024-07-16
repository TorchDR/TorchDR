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

from torch.testing import assert_close


from torchdr.eval import silhouette_samples, silhouette_score, admissible_LIST_METRICS
from torchdr.utils import pykeops

lst_types = [torch.float32, torch.float64]
if pykeops:
    lst_keops = [True, False]
    # lst_keops = [False]
else:
    lst_keops = [False]
DEVICE = "cpu"


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("keops", lst_keops)
def test_silhouette_score_euclidean(dtype, keops):
    # perfect silhouette
    n = 10
    I = torch.eye(n, device=DEVICE, dtype=dtype)
    y_I = torch.arange(n, device=DEVICE)
    ones = torch.ones(n, device=DEVICE, dtype=dtype)
    zeros = torch.zeros(n, device=DEVICE, dtype=dtype)

    y_I2 = []
    for i in range(n // 2):
        y_I2 += [i] * 2
    y_I2 = torch.tensor(y_I2, device=DEVICE)

    for metric in ["euclidean", "manhattan", "whatever"]:
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
