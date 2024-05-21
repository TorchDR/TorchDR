# -*- coding: utf-8 -*-
"""
Tests estimators for scikit-learn compatibility.
"""

# Author: Mathurin Massias
#         Hugues Van Assel
#
# License: BSD 3-Clause License

import pytest

from torchdr.neighbor_embedding import SNE, TSNE
from sklearn.utils.estimator_checks import check_estimator

estimator_list = [SNE, TSNE]


@pytest.mark.parametrize("klass", estimator_list)
def test_check_estimator(klass):
    check_estimator(klass(verbose=False))
