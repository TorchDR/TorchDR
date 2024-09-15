# -*- coding: utf-8 -*-
"""
Tests clustering estimators.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

from torchdr.clustering import KMeans as TorchKMeans
from torchdr.utils import pykeops


@pytest.fixture
def sample_data():
    """Create a simple dataset for clustering."""
    X = np.array(
        [[1.0, 2.0], [1.0, 4.0], [1.0, 0.0], [10.0, 2.0], [10.0, 4.0], [10.0, 0.0]]
    )
    return X


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("init_method", ["random", "k-means++"])
def test_kmeans_fit_predict(sample_data, dtype, init_method):
    """Test the KMeans fit_predict method with different dtypes and init methods."""
    X = torch.tensor(sample_data, dtype=dtype)
    kmeans = TorchKMeans(n_clusters=2, init=init_method, random_state=42)
    labels = kmeans.fit_predict(X)

    # Check that labels are of correct type and shape
    assert isinstance(labels, torch.Tensor)
    assert labels.shape[0] == X.shape[0]

    # Check that cluster_centers_ is set
    assert hasattr(kmeans, "cluster_centers_"), "cluster_centers_ not set after fitting"
    assert isinstance(kmeans.cluster_centers_, torch.Tensor)
    assert kmeans.cluster_centers_.dtype == dtype


@pytest.mark.parametrize("keops", [False, True])
def test_kmeans_keops(sample_data, keops):
    """Test KMeans with keops enabled or disabled."""
    if keops and not pykeops:
        pytest.skip("pykeops is not installed, skipping test with keops=True")
    X = torch.tensor(sample_data, dtype=torch.float64)
    kmeans = TorchKMeans(n_clusters=2, keops=keops, random_state=42)
    kmeans.fit(X)

    # Check that labels are set
    assert hasattr(kmeans, "labels_"), "labels_ not set after fitting"
    assert isinstance(kmeans.labels_, torch.Tensor)

    # Check that cluster_centers_ is set
    assert hasattr(kmeans, "cluster_centers_"), "cluster_centers_ not set after fitting"
    assert isinstance(kmeans.cluster_centers_, torch.Tensor)


def test_kmeans_sklearn_comparison(sample_data):
    """Compare TorchKMeans results with sklearn KMeans on a toy dataset."""
    # Set random state for reproducibility
    random_state = 42

    # Fit TorchKMeans
    torch_kmeans = TorchKMeans(
        n_clusters=2, init="k-means++", random_state=random_state
    )
    torch_kmeans.fit(sample_data)
    torch_labels = torch_kmeans.labels_.numpy()

    # Fit sklearn KMeans
    sklearn_kmeans = SklearnKMeans(
        n_clusters=2, init="k-means++", n_init=1, random_state=random_state
    )
    sklearn_kmeans.fit(sample_data)
    sklearn_labels = sklearn_kmeans.labels_

    # Since labels can be permuted, we need to check
    # if they are equal up to a permutation
    assert np.array_equal(torch_labels, sklearn_labels) or np.array_equal(
        torch_labels, 1 - sklearn_labels
    ), "Labels do not match sklearn KMeans"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_kmeans_predict(sample_data, dtype):
    """Test the KMeans predict method."""
    X = torch.tensor(sample_data, dtype=dtype)
    kmeans = TorchKMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    # Predict cluster labels
    predictions = kmeans.predict(X)

    # Check that predictions are correct type and shape
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape[0] == X.shape[0]

    # Predictions should match labels from fit
    assert torch.equal(
        predictions, kmeans.labels_
    ), "Predictions do not match labels from fit"


def test_kmeans_invalid_init():
    """Test that KMeans raises an error with invalid init method."""
    X = torch.randn(10, 2)
    with pytest.raises(ValueError, match="Unknown init method"):
        kmeans = TorchKMeans(n_clusters=2, init="invalid_method")
        kmeans.fit(X)
