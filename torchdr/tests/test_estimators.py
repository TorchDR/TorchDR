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

estimator_list = [SNE, TSNE, InfoTSNE]


@pytest.mark.parametrize("estimator", estimator_list)
def test_check_estimator(estimator):
    check_estimator(estimator(verbose=False, device="cpu", keops=False, max_iter=1))


@pytest.mark.parametrize("estimator", [SNEkhorn, TSNEkhorn])
def test_check_estimator_affinity_lr(estimator):
    check_estimator(
        estimator(
            verbose=False,
            device="cpu",
            keops=False,
            max_iter=1,
            lr_affinity_in=1e-3,
        )
    )
