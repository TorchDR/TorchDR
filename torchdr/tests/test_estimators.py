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
from torchdr.utils import pykeops

DEVICE = "cpu"


@pytest.mark.skipif(pykeops, reason="pykeops is available")
def test_keops_not_installed():
    with pytest.raises(ValueError, match="pykeops is not installed"):
        SNE(keops=True)


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
        estimator(verbose=False, device=DEVICE, keops=False, max_iter=1, **kwargs)
    )


@pytest.mark.skipif(pykeops, reason="pykeops is available")
def test_init_keops_error(monkeypatch):
    with pytest.raises(ValueError, match="pykeops is not installed"):
        TSNE(keops=True)


def test_init_verbose(capfd):
    TSNE(verbose=True)
    captured = capfd.readouterr()
    assert "Initializing" in captured.out
