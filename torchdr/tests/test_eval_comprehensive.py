"""
Comprehensive tests for the eval module.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
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
    kmeans_ari,
    neighborhood_preservation,
)
from torchdr.tests.utils import toy_dataset, iris_dataset
from torchdr.utils import pykeops
from torchdr.utils.faiss import faiss
from torchdr.distance import pairwise_distances, FaissConfig


# Test configuration
lst_types = ["float32", "float64"]
if pykeops:
    lst_backend = ["keops", None]
else:
    lst_backend = [None]

# Add faiss backend if available
if faiss:
    lst_backend.append("faiss")

DEVICE = "cpu"
if torch.cuda.is_available():
    CUDA_DEVICE = "cuda"
else:
    CUDA_DEVICE = None


# ============================================================================
# SILHOUETTE TESTS (keeping existing tests)
# ============================================================================


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "whatever"])
def test_silhouette_score_euclidean(dtype, backend, metric):
    """Test silhouette score with various metrics and backends."""
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
        # Test edge cases with isolated samples
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

        # Test with equidistant samples
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
    """Test silhouette score with precomputed distances."""
    n = 10
    Id = torch.eye(n, device=DEVICE, dtype=getattr(torch, dtype))
    CI = pairwise_distances(Id, Id, "euclidean")
    ones = torch.ones(n, device=DEVICE, dtype=getattr(torch, dtype))

    y_I2 = []
    for i in range(n // 2):
        y_I2 += [i] * 2
    y_I2 = torch.tensor(y_I2, device=DEVICE)

    # Test with precomputed distances
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

    # Test error cases
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
    """Test consistency with sklearn's silhouette score."""
    n = 100
    X, y = toy_dataset(n, dtype)

    score_torchdr = silhouette_score(X, y, None, metric, DEVICE, backend)
    score_sklearn = sk_silhouette_score(X, y, metric=metric)
    assert (score_torchdr - score_sklearn) ** 2 < 1e-5, (
        "Silhouette scores from torchdr and sklearn should be close."
    )

    # Generate noisy labels
    y_noise = y.copy()
    y_noise[: n // 2] = np.random.permutation(y_noise[: n // 2])

    score_torchdr_noise = silhouette_score(X, y_noise, None, metric, DEVICE, backend)
    score_sklearn_noise = sk_silhouette_score(X, y_noise, metric=metric)
    assert (score_torchdr_noise - score_sklearn_noise) ** 2 < 1e-5, (
        "Silhouette scores from torchdr and sklearn should be close on noised labels."
    )


@pytest.mark.parametrize("backend", lst_backend)
def test_silhouette_with_faiss_config(backend):
    """Test silhouette with FaissConfig object."""
    if backend == "faiss" and faiss:
        X, y = toy_dataset(100, "float32")

        # Test with FaissConfig
        config = FaissConfig()
        score_with_config = silhouette_score(X, y, backend=config)
        score_with_string = silhouette_score(X, y, backend="faiss")

        # Results should be similar (may have small differences due to implementation)
        assert abs(score_with_config - score_with_string) < 0.01


# ============================================================================
# K-MEANS ARI TESTS
# ============================================================================


@pytest.mark.skipif(not faiss, reason="FAISS not installed")
@pytest.mark.parametrize("dtype", lst_types)
def test_kmeans_ari_basic(dtype):
    """Test basic kmeans_ari functionality."""
    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        pytest.skip("torchmetrics not installed")

    X, y = toy_dataset(100, dtype)

    # Test with numpy arrays
    ari_score, pred_labels = kmeans_ari(X, y)
    assert isinstance(ari_score, float)
    assert isinstance(pred_labels, np.ndarray)
    assert -1 <= ari_score <= 1
    assert pred_labels.shape == (100,)

    # Test with torch tensors
    X_torch = torch.from_numpy(X)
    y_torch = torch.from_numpy(y)
    ari_score_torch, pred_labels_torch = kmeans_ari(X_torch, y_torch)
    assert isinstance(ari_score_torch, torch.Tensor)
    assert isinstance(pred_labels_torch, torch.Tensor)
    assert -1 <= ari_score_torch <= 1
    assert pred_labels_torch.shape == (100,)


@pytest.mark.skipif(not faiss, reason="FAISS not installed")
def test_kmeans_ari_n_clusters():
    """Test kmeans_ari with different n_clusters settings."""
    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        pytest.skip("torchmetrics not installed")

    X, y = iris_dataset("float32")

    # Test with automatic n_clusters (should use number of unique labels)
    ari_auto, _ = kmeans_ari(X, y)

    # Test with explicit n_clusters
    ari_explicit, _ = kmeans_ari(X, y, n_clusters=3)

    # Test with different n_clusters
    ari_diff, pred_labels = kmeans_ari(X, y, n_clusters=4)
    assert len(np.unique(pred_labels)) <= 4

    # Test error cases
    with pytest.raises(ValueError):
        kmeans_ari(X, y, n_clusters=0)

    with pytest.raises(ValueError):
        kmeans_ari(X, y, n_clusters=len(X) + 1)


@pytest.mark.skipif(not faiss, reason="FAISS not installed")
def test_kmeans_ari_reproducibility():
    """Test kmeans_ari reproducibility with random_state."""
    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        pytest.skip("torchmetrics not installed")

    X, y = toy_dataset(100, "float32")

    # Test reproducibility
    ari1, labels1 = kmeans_ari(X, y, random_state=42)
    ari2, labels2 = kmeans_ari(X, y, random_state=42)

    assert ari1 == ari2
    assert np.array_equal(labels1, labels2)

    # Test different random states give different results
    ari3, labels3 = kmeans_ari(X, y, random_state=123)
    # ARI might be similar but labels should differ
    assert not np.array_equal(labels1, labels3)


@pytest.mark.skipif(not faiss, reason="FAISS not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kmeans_ari_gpu():
    """Test kmeans_ari with GPU device."""
    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        pytest.skip("torchmetrics not installed")

    X, y = toy_dataset(100, "float32")
    X_cuda = torch.from_numpy(X).cuda()
    y_cuda = torch.from_numpy(y).cuda()

    # Test with GPU tensors
    ari_gpu, pred_labels_gpu = kmeans_ari(X_cuda, y_cuda, device="cuda")

    assert isinstance(ari_gpu, torch.Tensor)
    assert isinstance(pred_labels_gpu, torch.Tensor)
    assert pred_labels_gpu.device.type == "cuda"


@pytest.mark.skipif(not faiss, reason="FAISS not installed")
def test_kmeans_ari_perfect_clustering():
    """Test kmeans_ari with perfect clustering."""
    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        pytest.skip("torchmetrics not installed")

    # Create well-separated clusters
    np.random.seed(42)
    X1 = np.random.randn(50, 2) + np.array([0, 0])
    X2 = np.random.randn(50, 2) + np.array([10, 10])
    X3 = np.random.randn(50, 2) + np.array([-10, 10])
    X = np.vstack([X1, X2, X3]).astype("float32")
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)

    ari_score, pred_labels = kmeans_ari(X, y, n_clusters=3, nredo=5)

    # With well-separated clusters, ARI should be high
    assert ari_score > 0.8


@pytest.mark.skipif(not faiss, reason="FAISS not installed")
def test_kmeans_ari_niter_nredo():
    """Test kmeans_ari with different niter and nredo values."""
    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        pytest.skip("torchmetrics not installed")

    np.random.seed(42)
    X = np.random.randn(100, 5).astype("float32")
    y = np.array([0] * 50 + [1] * 50)

    # Test with different niter values
    ari_low_iter, _ = kmeans_ari(X, y, n_clusters=2, niter=5, random_state=42)
    ari_high_iter, _ = kmeans_ari(X, y, n_clusters=2, niter=50, random_state=42)

    # Both should produce valid ARI scores
    assert -1 <= ari_low_iter <= 1
    assert -1 <= ari_high_iter <= 1

    # Test with different nredo values
    ari_redo1, _ = kmeans_ari(X, y, n_clusters=2, nredo=1, random_state=42)
    ari_redo5, _ = kmeans_ari(X, y, n_clusters=2, nredo=5, random_state=42)

    assert -1 <= ari_redo1 <= 1
    assert -1 <= ari_redo5 <= 1


@pytest.mark.skipif(not faiss, reason="FAISS not installed")
def test_kmeans_ari_edge_cases():
    """Test kmeans_ari edge cases."""
    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        pytest.skip("torchmetrics not installed")

    # Edge case: n_clusters equals n_samples
    X = np.random.randn(5, 3).astype("float32")
    y = np.arange(5)
    ari_score, pred_labels = kmeans_ari(X, y, n_clusters=5)
    assert -1 <= ari_score <= 1
    assert len(pred_labels) == 5

    # Edge case: 2 clusters with minimal samples
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype="float32")
    y = np.array([0, 0, 1, 1])
    ari_score, pred_labels = kmeans_ari(X, y, n_clusters=2)
    assert ari_score > 0.5  # Should cluster well


# ============================================================================
# NEIGHBORHOOD PRESERVATION TESTS
# ============================================================================


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("backend", lst_backend)
def test_neighborhood_preservation_basic(dtype, backend):
    """Test basic neighborhood preservation functionality."""
    n = 50
    d_high = 10
    d_low = 2
    K = 5

    # Create random data
    X = torch.randn(n, d_high, dtype=getattr(torch, dtype))
    Z = torch.randn(n, d_low, dtype=getattr(torch, dtype))

    score = neighborhood_preservation(X, Z, K, backend=backend)

    assert 0 <= score <= 1
    assert isinstance(score, float) or isinstance(score, torch.Tensor)


@pytest.mark.parametrize("backend", lst_backend)
def test_neighborhood_preservation_perfect(backend):
    """Test neighborhood preservation with identical embeddings."""
    n = 30
    d = 5
    K = 5

    # When X and Z are identical, preservation should be perfect
    X = torch.randn(n, d)
    Z = X.clone()

    score = neighborhood_preservation(X, Z, K, backend=backend)
    assert_close(score, torch.tensor(1.0))


@pytest.mark.parametrize("backend", lst_backend)
def test_neighborhood_preservation_random(backend):
    """Test neighborhood preservation with random embeddings."""
    n = 50
    d_high = 10
    d_low = 2
    K = 10

    # Random embeddings should have low preservation
    torch.manual_seed(42)
    X = torch.randn(n, d_high)
    Z = torch.randn(n, d_low)

    score = neighborhood_preservation(X, Z, K, backend=backend)

    # Random embeddings should have low but non-zero preservation
    assert 0 < score < 0.5


def test_neighborhood_preservation_numpy():
    """Test neighborhood preservation with numpy arrays."""
    n = 30
    d_high = 10
    d_low = 2
    K = 5

    # Test with numpy arrays
    X_np = np.random.randn(n, d_high).astype("float32")
    Z_np = np.random.randn(n, d_low).astype("float32")

    score = neighborhood_preservation(X_np, Z_np, K)

    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_neighborhood_preservation_errors():
    """Test neighborhood preservation error handling."""
    X = torch.randn(50, 10)
    Z = torch.randn(30, 2)  # Different number of samples

    # Different number of samples
    with pytest.raises(ValueError):
        neighborhood_preservation(X, Z, K=5)

    # K too large
    X2 = torch.randn(10, 5)
    Z2 = torch.randn(10, 2)
    with pytest.raises(ValueError):
        neighborhood_preservation(X2, Z2, K=10)

    # K < 1
    with pytest.raises(ValueError):
        neighborhood_preservation(X2, Z2, K=0)


@pytest.mark.parametrize("K", [1, 5, 10, 20])
def test_neighborhood_preservation_different_k(K):
    """Test neighborhood preservation with different K values."""
    n = 50
    X = torch.randn(n, 10)
    Z = torch.randn(n, 2)

    score = neighborhood_preservation(X, Z, K)
    assert 0 <= score <= 1


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "sqeuclidean"])
def test_neighborhood_preservation_metrics(metric):
    """Test neighborhood preservation with different metrics."""
    X = torch.randn(30, 10)
    Z = torch.randn(30, 2)

    score = neighborhood_preservation(X, Z, K=5, metric=metric)
    assert 0 <= score <= 1


def test_neighborhood_preservation_with_faiss_config():
    """Test neighborhood preservation with FaissConfig."""
    if faiss:
        X = torch.randn(100, 50)
        Z = torch.randn(100, 2)

        config = FaissConfig()
        score = neighborhood_preservation(X, Z, K=10, backend=config)
        assert 0 <= score <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_neighborhood_preservation_gpu():
    """Test neighborhood preservation on GPU."""
    X = torch.randn(50, 10).cuda()
    Z = torch.randn(50, 2).cuda()

    score = neighborhood_preservation(X, Z, K=5, device="cuda")
    assert 0 <= score <= 1


def test_neighborhood_preservation_dimensionality_reduction():
    """Test neighborhood preservation with actual DR scenario."""
    from sklearn.decomposition import PCA

    # Generate data with structure
    X, _ = toy_dataset(100, "float32")

    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    # PCA should preserve some neighborhood structure
    score = neighborhood_preservation(X, Z, K=10)

    # PCA preserves global structure, so should have reasonable preservation
    assert score > 0.3


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.parametrize("backend", lst_backend)
def test_eval_functions_consistency(backend):
    """Test that all eval functions work with same data and backends."""
    X, y = toy_dataset(50, "float32")

    # All functions should work with the same data
    sil_score = silhouette_score(X, y, backend=backend)
    assert -1 <= sil_score <= 1

    if faiss:
        try:
            import torchmetrics  # noqa: F401

            ari_score, _ = kmeans_ari(X, y)
            assert -1 <= ari_score <= 1
        except ImportError:
            pass

    # Create a simple embedding for neighborhood preservation
    Z = X[:, :2]  # Use first 2 dimensions as embedding
    np_score = neighborhood_preservation(X, Z, K=5, backend=backend)
    assert 0 <= np_score <= 1


def test_eval_functions_type_preservation():
    """Test that eval functions preserve input types."""
    X_np, y_np = toy_dataset(50, "float32")
    X_torch = torch.from_numpy(X_np)
    y_torch = torch.from_numpy(y_np)

    # Test silhouette
    sil_np = silhouette_score(X_np, y_np)
    assert isinstance(sil_np, (float, np.floating))

    sil_torch = silhouette_score(X_torch, y_torch)
    assert isinstance(sil_torch, torch.Tensor)

    # Test neighborhood preservation
    Z_np = X_np[:, :2]
    Z_torch = X_torch[:, :2]

    np_score_np = neighborhood_preservation(X_np, Z_np, K=5)
    assert isinstance(np_score_np, (float, np.floating))

    np_score_torch = neighborhood_preservation(X_torch, Z_torch, K=5)
    assert isinstance(np_score_torch, torch.Tensor)


# ============================================================================
# BENCHMARKING TESTS (optional, for performance)
# ============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("n_samples", [100, 500, 1000])
@pytest.mark.parametrize("backend", lst_backend)
def test_eval_performance(n_samples, backend):
    """Benchmark eval functions with different data sizes."""
    import time

    X = np.random.randn(n_samples, 50).astype("float32")
    y = np.random.randint(0, 10, n_samples)
    Z = np.random.randn(n_samples, 2).astype("float32")

    # Time silhouette score
    start = time.time()
    _ = silhouette_score(X, y, backend=backend)
    sil_time = time.time() - start

    # Time neighborhood preservation
    start = time.time()
    _ = neighborhood_preservation(X, Z, K=10, backend=backend)
    np_time = time.time() - start

    print(f"\nBackend: {backend}, N={n_samples}")
    print(f"Silhouette time: {sil_time:.3f}s")
    print(f"Neighborhood preservation time: {np_time:.3f}s")

    # Just ensure they complete without error
    assert sil_time < 60  # Should complete within 60 seconds
    assert np_time < 60
