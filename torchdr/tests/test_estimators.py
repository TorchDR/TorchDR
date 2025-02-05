"""
Tests estimators for scikit-learn compatibility.
"""

# Author: Mathurin Massias
#         Hugues Van Assel
#
# License: BSD 3-Clause License

import pytest
from sklearn.utils.estimator_checks import check_estimator

from torchdr.neighbor_embedding import SNE, TSNE, InfoTSNE, LargeVis, TSNEkhorn

DEVICE = "cpu"


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (SNE, {}),
        (TSNE, {}),
        (InfoTSNE, {}),
        (TSNEkhorn, {"lr_affinity_in": 1e-3}),
        (LargeVis, {}),
    ],
)
def test_check_estimator(estimator, kwargs):
    check_estimator(
        estimator(verbose=False, device=DEVICE, backend=None, max_iter=1, **kwargs)
    )


def test_init_verbose(capfd):
    TSNE(verbose=True)
    captured = capfd.readouterr()
    assert "Initializing" in captured.out
