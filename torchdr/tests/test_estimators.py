# -*- coding: utf-8 -*-
"""
Tests estimators for scikit-learn compatibility.
"""

# Author: Mathurin Massias
#         Hugues Van Assel
#
# License: BSD 3-Clause License

import pytest

from torchdr.neighbor_embedding import (
    SNE,
    TSNE,
    InfoTSNE,
    SNEkhorn,
    TSNEkhorn,
    LargeVis,
)
from torchdr.utils import pykeops
from sklearn.utils.estimator_checks import check_estimator

DEVICE = "cpu"


@pytest.mark.skipif(pykeops, reason="pykeops is available")
def test_keops_not_installed():
    with pytest.raises(ValueError, match="KeOps is not available"):
        SNE(keops=True)


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (SNE, {}),
        (TSNE, {}),
        (InfoTSNE, {}),
        (SNEkhorn, {"lr_affinity_in": 1e-3}),
        (TSNEkhorn, {"lr_affinity_in": 1e-3}),
        (LargeVis, {}),
    ],
)
def test_check_estimator(estimator, kwargs):
    check_estimator(
        estimator(verbose=False, device=DEVICE, keops=False, max_iter=1, **kwargs)
    )
