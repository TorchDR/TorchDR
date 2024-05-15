# -*- coding: utf-8 -*-
"""
Tests estimators for scikit-learn compatibility.
"""

# Author: Mathurin Massias
#
# License: BSD 3-Clause License

import pytest

from torchdr.neighbor_embedding import SNE, TSNE
from sklearn.utils.estimator_checks import check_estimator


@pytest.mark.parametrize("klass", [SNE, TSNE])
def test_check_estimator(klass):
    check_estimator(klass())