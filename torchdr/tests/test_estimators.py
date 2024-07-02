# -*- coding: utf-8 -*-
"""
Tests estimators for scikit-learn compatibility.
"""

# Author: Mathurin Massias
#         Hugues Van Assel
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr.neighbor_embedding import (
    SNE,
    TSNE,
    InfoTSNE,
    SNEkhorn,
    TSNEkhorn,
    LargeVis,
)
from sklearn.utils.estimator_checks import check_estimator

DEVICE = "cpu"


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (SNE, {}),
        (TSNE, {}),
        (InfoTSNE, {}),
        (SNEkhorn, {"lr_affinity_in": 1e-3, "max_iter_affinity_out": 1}),
        (TSNEkhorn, {"lr_affinity_in": 1e-3, "max_iter_affinity_out": 1}),
        (LargeVis, {}),
    ],
)
def test_check_estimator(estimator, kwargs):
    check_estimator(
        estimator(verbose=False, device=DEVICE, keops=False, max_iter=1, **kwargs)
    )
