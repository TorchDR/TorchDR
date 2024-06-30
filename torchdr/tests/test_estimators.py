# -*- coding: utf-8 -*-
"""
Tests estimators for scikit-learn compatibility.
"""

# Author: Mathurin Massias
#         Hugues Van Assel
#
# License: BSD 3-Clause License

import pytest

from torchdr.neighbor_embedding import SNE, TSNE, InfoTSNE, SNEkhorn, TSNEkhorn
from sklearn.utils.estimator_checks import check_estimator


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (SNE, {}),
        (TSNE, {}),
        (InfoTSNE, {}),
        (SNEkhorn, {"lr_affinity_in": 1e-3}),
        (TSNEkhorn, {"lr_affinity_in": 1e-3}),
    ],
)
def test_check_estimator(estimator, kwargs):
    check_estimator(
        estimator(verbose=False, device="cpu", keops=False, max_iter=1, **kwargs)
    )
