# -*- coding: utf-8 -*-
"""
Tests for functions in eval module.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: BSD 3-Clause License

import torch
import pytest

from torch.testing import assert_close


from torchdr.eval import silhouette_samples, silhouette_score
from torchdr.utils import pykeops, LIST_METRICS

lst_types = [torch.double, torch.float]
if pykeops:
    lst_keops = [True, False]
else:
    lst_keops = [False]
DEVICE = "cpu"


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("keops", lst_keops)
def test_silhouette_score(dtype, keops):
    # perfect silhouette
    n = 10
    I = torch.eye(n, device=DEVICE, dtype=dtype)
    y = torch.arange(n, device=DEVICE)
    ones = torch.ones(n, device=DEVICE, dtype=dtype)
    for metric in LIST_METRICS:
        coeffs = silhouette_samples(I, y, None, metric, None, keops)
        assert_close(coeffs, ones)

        weighted_coeffs = silhouette_samples(I, y, ones / n, metric, DEVICE, keops)
        assert_close(coeffs, weighted_coeffs)

        score = silhouette_score(I, y, None, metric, DEVICE, keops)
        assert_close(coeffs.mean(), score)
