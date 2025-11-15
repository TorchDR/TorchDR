"""
Tests for functions in eval module.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: BSD 3-Clause License

import warnings

import numpy as np
import pytest
import torch
from sklearn.metrics import silhouette_score as sk_silhouette_score
from torch.testing import assert_close

from torchdr.eval import (
    admissible_LIST_METRICS,
    silhouette_samples,
    silhouette_score,
    knn_label_accuracy,
)
from torchdr.tests.utils import toy_dataset
from torchdr.utils import pykeops
from torchdr.distance import pairwise_distances

lst_types = ["float32", "float64"]
if pykeops:
    lst_backend = ["keops", None]
else:
    lst_backend = [None]
DEVICE = "cpu"


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "whatever"])
def test_silhouette_score_euclidean(dtype, backend, metric):
    # perfect silhouette
    n = 10
    Id = torch.eye(n, device=DEVICE, dtype=getattr(torch, dtype))
    y_I = torch.arange(n, device=DEVICE)
    ones = torch.ones(n, device=DEVICE, dtype=getattr(torch, dtype))
    zeros = torch.zeros(n, device=DEVICE, dtype=getattr(torch, dtype))

    y_I2 = []
    for i in range(n // 2):
        y_I2 += [i] * 2
    y_I2 = torch.tensor(y_I2, device=DEVICE)

    if metric in admissible_LIST_METRICS:
        # tests edge cases with isolated samples
        with warnings.catch_warnings(record=True) as w:
            coeffs = silhouette_samples(Id, y_I, None, metric, None, backend, True)
            assert issubclass(w[-1].category, UserWarning)

        assert_close(coeffs, ones)
        weighted_coeffs = silhouette_samples(
            Id, y_I, ones / n, metric, DEVICE, backend, False
        )
        assert_close(coeffs, weighted_coeffs)
        score = silhouette_score(Id, y_I, None, metric, DEVICE, backend)
        assert_close(coeffs.mean(), score)
        sampled_score = silhouette_score(Id, y_I, None, metric, DEVICE, backend, n)
        assert_close(score, sampled_score)

        # tests with equidistant samples
        coeffs_2 = silhouette_samples(Id, y_I2, ones / n, metric, None, backend)
        assert_close(coeffs_2, zeros)

        weighted_coeffs_2 = silhouette_samples(
            Id, y_I2, None, metric, DEVICE, backend, False
        )
        assert_close(coeffs_2, weighted_coeffs_2)

        score_2 = silhouette_score(Id, y_I2, None, metric, DEVICE, backend)
        assert_close(coeffs_2.mean(), score_2)

    else:
        with pytest.raises(ValueError):
            _ = silhouette_samples(Id, y_I, None, metric, None, backend, True)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
def test_silhouette_score_precomputed(dtype, backend):
    # perfect silhouette
    n = 10
    Id = torch.eye(n, device=DEVICE, dtype=getattr(torch, dtype))
    CI = pairwise_distances(Id, Id, "euclidean")
    ones = torch.ones(n, device=DEVICE, dtype=getattr(torch, dtype))

    y_I2 = []
    for i in range(n // 2):
        y_I2 += [i] * 2
    y_I2 = torch.tensor(y_I2, device=DEVICE)

    # tests edge cases with isolated samples
    coeffs_pre = silhouette_samples(CI, y_I2, None, "precomputed", None, backend, True)
    coeffs = silhouette_samples(Id, y_I2, None, "euclidean", None, backend, True)

    assert_close(coeffs_pre, coeffs)
    weighted_coeffs_pre = silhouette_samples(
        CI, y_I2, ones / n, "precomputed", DEVICE, backend, False
    )
    assert_close(coeffs_pre, weighted_coeffs_pre)
    score_pre = silhouette_score(CI, y_I2, None, "precomputed", DEVICE, backend)
    assert_close(coeffs_pre.mean(), score_pre)
    sampled_score_pre = silhouette_score(
        CI, y_I2, None, "precomputed", DEVICE, backend, n
    )
    assert_close(score_pre, sampled_score_pre)

    # catch errors
    with pytest.raises(ValueError):
        _ = silhouette_samples(
            CI[:, :-2], y_I2, None, "precomputed", None, backend, True
        )

    with pytest.raises(ValueError):
        _ = silhouette_score(CI[:, :-2], y_I2, None, "precomputed", None, backend, n)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_consistency_sklearn(dtype, backend, metric):
    n = 100
    X, y = toy_dataset(n, dtype)

    score_torchdr = silhouette_score(X, y, None, metric, DEVICE, backend)
    score_sklearn = sk_silhouette_score(X, y, metric=metric)
    assert (score_torchdr - score_sklearn) ** 2 < 1e-5, (
        "Silhouette scores from torchdr and sklearn should be close."
    )

    # Generate y_noise by permuting half of the indices in y randomly
    y_noise = y.copy()
    y_noise[: n // 2] = np.random.permutation(y_noise[: n // 2])

    score_torchdr_noise = silhouette_score(X, y_noise, None, metric, DEVICE, backend)
    score_sklearn_noise = sk_silhouette_score(X, y_noise, metric=metric)
    assert (score_torchdr_noise - score_sklearn_noise) ** 2 < 1e-5, (
        "Silhouette scores from torchdr and sklearn should be close on noised labels."
    )


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
@pytest.mark.parametrize("k", [5, 10])
def test_knn_label_accuracy_perfect(dtype, backend, k):
    n_per_class = 50
    n_classes = 3
    n = n_per_class * n_classes

    X = torch.randn(n, 10, device=DEVICE, dtype=getattr(torch, dtype))
    labels = torch.repeat_interleave(
        torch.arange(n_classes, device=DEVICE), n_per_class
    )

    for i in range(n_classes):
        start = i * n_per_class
        end = (i + 1) * n_per_class
        X[start:end] += i * 10

    accuracy = knn_label_accuracy(X, labels, k=k, backend=backend, device=DEVICE)

    assert accuracy > 0.95, (
        f"Expected high accuracy for well-separated classes, got {accuracy}"
    )


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
def test_knn_label_accuracy_random(dtype, backend):
    n = 100
    k = 10

    X = torch.randn(n, 20, device=DEVICE, dtype=getattr(torch, dtype))
    labels = torch.randint(0, 5, (n,), device=DEVICE)

    accuracy = knn_label_accuracy(X, labels, k=k, backend=backend, device=DEVICE)

    assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
def test_knn_label_accuracy_per_sample(dtype, backend):
    n = 100
    k = 10

    X = torch.randn(n, 20, device=DEVICE, dtype=getattr(torch, dtype))
    labels = torch.randint(0, 3, (n,), device=DEVICE)

    accuracies = knn_label_accuracy(
        X, labels, k=k, backend=backend, device=DEVICE, return_per_sample=True
    )

    assert accuracies.shape == (n,), f"Expected shape ({n},), got {accuracies.shape}"
    assert torch.all((accuracies >= 0) & (accuracies <= 1)), (
        "All accuracies must be in [0, 1]"
    )

    mean_accuracy = knn_label_accuracy(
        X, labels, k=k, backend=backend, device=DEVICE, return_per_sample=False
    )
    assert_close(mean_accuracy, accuracies.mean(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", lst_types)
def test_knn_label_accuracy_numpy(dtype):
    n = 100
    k = 10

    X_np = np.random.randn(n, 20).astype(dtype)
    labels_np = np.random.randint(0, 3, n)

    accuracy = knn_label_accuracy(X_np, labels_np, k=k, backend=None)

    assert isinstance(accuracy, (float, np.floating)), (
        "Should return numpy float for numpy inputs"
    )
    assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"

    accuracies = knn_label_accuracy(
        X_np, labels_np, k=k, backend=None, return_per_sample=True
    )
    assert isinstance(accuracies, np.ndarray), (
        "Should return numpy array for numpy inputs"
    )


def test_knn_label_accuracy_errors():
    n = 100
    X = torch.randn(n, 20)
    labels = torch.randint(0, 3, (n,))

    with pytest.raises(ValueError, match="k must be at least 1"):
        knn_label_accuracy(X, labels, k=0)

    with pytest.raises(ValueError, match="k.*must be less than number of samples"):
        knn_label_accuracy(X, labels, k=n)

    with pytest.raises(ValueError, match="must have same number of samples"):
        knn_label_accuracy(X, labels[:-10], k=10)
