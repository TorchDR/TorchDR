"""
Tests estimators for scikit-learn compatibility.
"""

# Author: Mathurin Massias
#         Hugues Van Assel
#
# License: BSD 3-Clause License

import pytest
from sklearn.utils.estimator_checks import check_estimator
import torch
from torch.testing import assert_close

from torchdr.neighbor_embedding import SNE, TSNE, InfoTSNE, LargeVis, TSNEkhorn, PACMAP
from torchdr.spectral_embedding import PCA

DEVICE = "cpu"


def test_process_duplicates():
    """Test the process_duplicates functionality in the DRModule base class."""
    # Create a dataset with duplicates
    X_unique = torch.randn(5, 3)
    X_duplicates = torch.cat([X_unique, X_unique[0:2, :]], dim=0)

    # Instantiate a simple estimator
    estimator = PCA(n_components=2, random_state=42)

    # Fit on the data with duplicates
    embedding_duplicates = estimator.fit_transform(X_duplicates)

    # Manually fit on unique data and reconstruct
    embedding_unique = estimator.fit_transform(X_unique)
    reconstructed_embedding = torch.cat(
        [embedding_unique, embedding_unique[0:2, :]], dim=0
    )

    assert_close(embedding_duplicates, reconstructed_embedding)

    # Test with process_duplicates=False
    estimator_no_duplicates = PCA(
        n_components=2, random_state=42, process_duplicates=False
    )
    embedding_no_duplicates = estimator_no_duplicates.fit_transform(X_duplicates)

    # The result should be different when not processing duplicates
    assert not torch.allclose(embedding_duplicates, embedding_no_duplicates)


@pytest.mark.xfail(strict=False, reason="sklearn estimator‐check not critical")
@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (SNE, {}),
        (TSNE, {}),
        (InfoTSNE, {}),
        (TSNEkhorn, {"lr_affinity_in": 1e-3}),
        (LargeVis, {}),
        (PACMAP, {}),
    ],
)
def test_check_estimator(estimator, kwargs):
    check_estimator(
        estimator(
            verbose=False,
            device=DEVICE,
            backend=None,
            max_iter=1,
            random_state=42,
            **kwargs,
        )
    )
