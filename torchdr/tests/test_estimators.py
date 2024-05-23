# -*- coding: utf-8 -*-
"""
Tests estimators for scikit-learn compatibility.
"""

# Author: Mathurin Massias
#         Hugues Van Assel
#
# License: BSD 3-Clause License

import pytest

from torchdr.neighbor_embedding import SNE, TSNE, InfoTSNE
from sklearn.utils.estimator_checks import check_estimator

estimator_list = [SNE, TSNE, InfoTSNE]


@pytest.mark.parametrize("estimator", estimator_list)
def test_check_estimator(estimator):
    check_estimator(estimator(verbose=False, device="cpu", keops=False, max_iter=1))
