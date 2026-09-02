"""Tests for DataLoader support in distance computation."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import gc
import weakref

import numpy as np
import pytest
import torch
from torch.testing import assert_close
from torch.utils.data import DataLoader, TensorDataset

from torchdr.distance import (
    pairwise_distances,
    pairwise_distances_faiss_from_dataloader,
    FaissConfig,
)
from torchdr.distance.faiss import (
    _DATALOADER_METADATA_CACHE,
    _BatchStager,
    _cache_dataloader_metadata,
    _reserve_index_capacity,
    _stream,
    get_dataloader_metadata,
)
from torchdr.utils import faiss


# Skip all tests if faiss is not available (faiss is False or None when not installed)
pytestmark = pytest.mark.skipif(
    faiss is None or faiss is False, reason="faiss not installed"
)

# A DataLoader is streamed into a GPU index where one is available, so its
# distances are compared against a CPU index. Both compute exact L2 from
# ``|x|^2 + |y|^2 - 2<x, y>``, and the cancellation leaves about 1e-5 on a
# squared self-distance; the square root of that residual is a few 1e-3.
EUCLIDEAN_ATOL = 1e-2


class TestDataLoaderDistances:
    """Test pairwise_distances with DataLoader input."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        n_samples = 1000
        n_features = 32
        X = torch.randn(n_samples, n_features)
        return X

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=100, shuffle=False)

    def test_dataloader_basic(self, sample_data, dataloader):
        """Test that DataLoader produces same results as tensor input."""
        k = 10

        # Compute with tensor
        dist_tensor, idx_tensor = pairwise_distances(
            sample_data, k=k, backend="faiss", return_indices=True
        )

        # Compute with DataLoader
        dist_dl, idx_dl = pairwise_distances(dataloader, k=k, return_indices=True)

        # Results should match
        assert_close(dist_tensor, dist_dl, rtol=1e-4, atol=EUCLIDEAN_ATOL)
        assert torch.equal(idx_tensor, idx_dl)

    def test_dataloader_exclude_diag(self, sample_data, dataloader):
        """Test exclude_diag with DataLoader."""
        k = 10

        # Compute with tensor
        dist_tensor, idx_tensor = pairwise_distances(
            sample_data, k=k, backend="faiss", exclude_diag=True, return_indices=True
        )

        # Compute with DataLoader
        dist_dl, idx_dl = pairwise_distances(
            dataloader, k=k, exclude_diag=True, return_indices=True
        )

        # Results should match
        assert_close(dist_tensor, dist_dl, rtol=1e-4, atol=1e-4)
        assert torch.equal(idx_tensor, idx_dl)

    @pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean"])
    def test_dataloader_metrics(self, sample_data, dataloader, metric):
        """Test different metrics with DataLoader."""
        k = 10

        # Compute with tensor
        dist_tensor, idx_tensor = pairwise_distances(
            sample_data, k=k, metric=metric, backend="faiss", return_indices=True
        )

        # Compute with DataLoader
        dist_dl, idx_dl = pairwise_distances(
            dataloader, k=k, metric=metric, return_indices=True
        )

        # Results should match
        atol = EUCLIDEAN_ATOL if metric == "euclidean" else 1e-4
        assert_close(dist_tensor, dist_dl, rtol=1e-4, atol=atol)

    def test_dataloader_with_config(self, sample_data, dataloader):
        """Test DataLoader with FaissConfig."""
        k = 10
        config = FaissConfig(index_type="Flat")

        # Compute with DataLoader and config
        dist_dl, idx_dl = pairwise_distances(
            dataloader, k=k, backend=config, return_indices=True
        )

        # Should produce valid results
        assert dist_dl.shape == (len(sample_data), k)
        assert idx_dl.shape == (len(sample_data), k)

    def test_dataloader_requires_k(self, dataloader):
        """Test that DataLoader requires k parameter."""
        with pytest.raises(ValueError, match="k cannot be None"):
            pairwise_distances(dataloader, k=None)

    def test_dataloader_no_cross_distance(self, sample_data, dataloader):
        """Test that DataLoader doesn't support cross-distance."""
        Y = torch.randn(100, 32)
        with pytest.raises(ValueError, match="Y must be None"):
            pairwise_distances(dataloader, Y=Y, k=10)

    def test_dataloader_unsupported_backend(self, dataloader):
        """Test that DataLoader raises error for non-FAISS backends."""
        with pytest.raises(ValueError, match="only supports FAISS backend"):
            pairwise_distances(dataloader, k=10, backend="keops")

    def test_dataloader_results_follow_batch_device(self, sample_data, dataloader):
        """Results come back on the device the batches live on."""
        dist, idx = pairwise_distances(dataloader, k=10, return_indices=True)

        assert dist.device == sample_data.device
        assert idx.device == sample_data.device

    @pytest.mark.parametrize(
        "n_samples,batch_size", [(1000, 300), (997, 256), (513, 512)]
    )
    def test_dataloader_non_divisible_final_batch(self, n_samples, batch_size):
        """A short final batch is handled like any other, down to a single row."""
        torch.manual_seed(42)
        X = torch.randn(n_samples, 32)
        dl = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

        dist_ref, idx_ref = pairwise_distances(
            X, k=10, backend="faiss", return_indices=True
        )
        dist_dl, idx_dl = pairwise_distances(dl, k=10, return_indices=True)

        assert dist_dl.shape == (n_samples, 10)
        assert_close(dist_ref, dist_dl, rtol=1e-3, atol=EUCLIDEAN_ATOL)
        assert torch.equal(idx_ref, idx_dl)

    def test_dataloader_different_batch_sizes(self, sample_data):
        """Test DataLoader with different batch sizes produces same results."""
        k = 10

        # Reference with full tensor
        dist_ref, idx_ref = pairwise_distances(
            sample_data, k=k, backend="faiss", return_indices=True
        )

        # Test different batch sizes
        for batch_size in [50, 100, 256, 500]:
            dataset = TensorDataset(sample_data)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            dist_dl, idx_dl = pairwise_distances(dl, k=k, return_indices=True)

            # Use looser tolerance due to float32 precision differences
            assert_close(dist_ref, dist_dl, rtol=1e-3, atol=1e-2)
            assert torch.equal(idx_ref, idx_dl)


class TestDataLoaderDirectFunction:
    """Test pairwise_distances_faiss_from_dataloader directly."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(500, 16)

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=50, shuffle=False)

    def test_direct_function_basic(self, sample_data, dataloader):
        """Test direct function call."""
        k = 5

        dist, idx = pairwise_distances_faiss_from_dataloader(
            dataloader, k=k, metric="sqeuclidean"
        )

        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)

    def test_direct_function_angular(self, sample_data):
        """Test angular metric."""
        # Normalize for angular metric
        sample_data = sample_data / sample_data.norm(dim=1, keepdim=True)
        dataset = TensorDataset(sample_data)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=False)

        k = 5
        dist, idx = pairwise_distances_faiss_from_dataloader(
            dataloader, k=k, metric="angular"
        )

        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDataLoaderGPU:
    """Test DataLoader with GPU computation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(500, 32)

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=100, shuffle=False)

    def test_dataloader_cuda(self, sample_data, dataloader):
        """Test DataLoader with CUDA device."""
        k = 10

        # Compute on CPU first for reference
        dist_cpu, idx_cpu = pairwise_distances(
            sample_data, k=k, backend="faiss", device="cpu", return_indices=True
        )

        # Compute with DataLoader on GPU
        dist_gpu, idx_gpu = pairwise_distances(
            dataloader, k=k, device="cuda", return_indices=True
        )

        # Move GPU results to CPU for comparison
        dist_gpu = dist_gpu.cpu()
        idx_gpu = idx_gpu.cpu()

        assert_close(dist_cpu, dist_gpu, rtol=1e-4, atol=EUCLIDEAN_ATOL)
        assert torch.equal(idx_cpu, idx_gpu)

    def test_dataloader_cuda_batches(self, sample_data):
        """Batches already on the GPU are searched and answered in place."""
        k = 10
        X_gpu = sample_data.cuda()
        dl = DataLoader(TensorDataset(X_gpu), batch_size=100, shuffle=False)

        dist_cpu, idx_cpu = pairwise_distances(
            sample_data, k=k, backend="faiss", device="cpu", return_indices=True
        )
        dist_gpu, idx_gpu = pairwise_distances(dl, k=k, return_indices=True)

        assert dist_gpu.device.type == "cuda"
        assert idx_gpu.device.type == "cuda"
        assert_close(dist_cpu, dist_gpu.cpu(), rtol=1e-4, atol=EUCLIDEAN_ATOL)
        assert torch.equal(idx_cpu, idx_gpu.cpu())

    def test_dataloader_gpu_avoids_numpy(self, dataloader, monkeypatch):
        """Host batches reach a GPU index without a NumPy round trip."""

        def forbidden(self, *args, **kwargs):
            raise AssertionError("batch was converted to NumPy")

        monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
        dist, idx = pairwise_distances(
            dataloader, k=10, device="cuda", return_indices=True
        )

        assert dist.device.type == "cuda"
        assert idx.device.type == "cuda"


class TestBatchStaging:
    """Test how DataLoader batches are handed to FAISS."""

    def test_stager_makes_batches_contiguous_float32(self):
        """A non-contiguous float64 batch reaches FAISS as contiguous float32."""
        batch = torch.randn(8, 6, dtype=torch.float64).T.contiguous().T[:, :4]
        assert not batch.is_contiguous()

        staged = _BatchStager(torch.device("cpu"))(batch)

        assert isinstance(staged, torch.Tensor)
        assert staged.dtype == torch.float32
        assert staged.is_contiguous()
        assert_close(staged, batch.float())

    def test_stager_numpy_fallback(self):
        """Builds without the torch wrappers still get float32 arrays."""
        batch = torch.randn(8, 4, dtype=torch.float64)

        staged = _BatchStager(torch.device("cpu"), to_numpy=True)(batch)

        assert isinstance(staged, np.ndarray)
        assert staged.dtype == np.float32

    def test_stream_regroups_batches(self):
        """Small batches are merged and large ones split, in order."""
        stage = _BatchStager(torch.device("cpu"))
        batches = [torch.full((3, 2), float(i)) for i in range(4)]

        groups = list(_stream(batches, stage, group_rows=6))

        assert [len(g) for g in groups] == [6, 6]
        assert_close(torch.cat(groups, dim=0), torch.cat(batches, dim=0))

        groups = list(_stream([torch.arange(20.0).reshape(10, 2)], stage, group_rows=4))
        assert [len(g) for g in groups] == [4, 4, 2]

    def test_stream_can_pass_batches_through(self):
        """A None target hands every batch over as it arrives."""
        stage = _BatchStager(torch.device("cpu"))
        batches = [torch.zeros(3, 2), torch.zeros(3, 2)]

        groups = list(_stream(batches, stage, group_rows=None))

        assert [len(g) for g in groups] == [3, 3]

    def test_stream_batch_size_must_be_positive(self):
        """A nonsense call size is rejected before any work is done."""
        X = torch.randn(64, 8)
        dl = DataLoader(TensorDataset(X), batch_size=16, shuffle=False)

        with pytest.raises(ValueError, match="stream_batch_size"):
            pairwise_distances(
                dl, k=5, backend=FaissConfig(stream_batch_size=0), return_indices=True
            )

    @pytest.mark.parametrize("stream_batch_size", [7, 64, 4096])
    def test_stream_batch_size_does_not_change_neighbors(self, stream_batch_size):
        """Regrouping is invisible in the neighbors it produces.

        FAISS picks its exact-search kernel from the number of queries in a
        call, so the distances move by the usual cancellation residual, but the
        neighbors they rank do not.
        """
        torch.manual_seed(42)
        X = torch.randn(300, 16)
        dl = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)

        dist_ref, idx_ref = pairwise_distances(dl, k=5, return_indices=True)
        dist, idx = pairwise_distances(
            dl,
            k=5,
            backend=FaissConfig(stream_batch_size=stream_batch_size),
            return_indices=True,
        )

        assert torch.equal(idx_ref, idx)
        assert_close(dist_ref, dist, rtol=1e-4, atol=EUCLIDEAN_ATOL)

    def test_reserve_capacity_ignores_unsupported_index(self):
        """An index without a reservation call is left untouched."""

        class NoReserve:
            pass

        _reserve_index_capacity(NoReserve(), 1000)

    def test_reserve_capacity_uses_available_call(self):
        """Whichever reservation call the build exposes is the one used."""

        class ReserveVecs:
            def __init__(self):
                self.reserved = None

            def reserveVecs(self, n):  # FAISS spelling
                self.reserved = n

        class ReserveMemory:
            def __init__(self):
                self.reserved = None

            def reserveMemory(self, n):  # FAISS spelling
                self.reserved = n

        flat, ivf = ReserveVecs(), ReserveMemory()
        _reserve_index_capacity(flat, 1000)
        _reserve_index_capacity(ivf, 1000)

        assert flat.reserved == 1000
        assert ivf.reserved == 1000

    def test_reserve_capacity_tolerates_rejected_request(self):
        """A build that refuses the reservation just grows on demand."""

        class Refuses:
            def reserveVecs(self, n):  # FAISS spelling
                raise RuntimeError("not supported for this index")

        _reserve_index_capacity(Refuses(), 1000)


class TestDataLoaderOptimizations:
    """Test DataLoader optimizations (metadata caching, shuffle validation)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(500, 32)

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=100, shuffle=False)

    def test_metadata_caching(self, sample_data, dataloader):
        """Test that metadata is cached after first pairwise_distances call."""
        k = 10

        # Before computation, metadata should not exist
        metadata_before = get_dataloader_metadata(dataloader)
        assert metadata_before is None

        # Compute k-NN (this should cache metadata)
        dist, idx = pairwise_distances(dataloader, k=k, return_indices=True)

        # After computation, metadata should be cached
        metadata_after = get_dataloader_metadata(dataloader)
        assert metadata_after is not None
        assert metadata_after["n_samples"] == len(sample_data)
        assert metadata_after["n_features"] == sample_data.shape[1]
        assert metadata_after["dtype"] == sample_data.dtype

        # Verify results are correct
        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)

    def test_metadata_cache_releases_destroyed_dataloader(self, sample_data):
        """Cached metadata should not outlive its DataLoader."""
        dataloader = DataLoader(
            TensorDataset(sample_data), batch_size=100, shuffle=False
        )
        metadata = {"cache_test_marker": object()}
        _cache_dataloader_metadata(dataloader, metadata)

        dataloader_ref = weakref.ref(dataloader)
        assert metadata in _DATALOADER_METADATA_CACHE.values()

        del dataloader
        gc.collect()

        assert dataloader_ref() is None
        assert metadata not in _DATALOADER_METADATA_CACHE.values()

    def test_shuffle_validation_sequential_sampler(self, sample_data):
        """Test that sequential sampler (shuffle=False) is accepted."""
        dataset = TensorDataset(sample_data)
        dl = DataLoader(dataset, batch_size=100, shuffle=False)

        # Should work without error
        dist, idx = pairwise_distances(dl, k=10, return_indices=True)
        assert dist.shape == (len(sample_data), 10)

    def test_shuffle_validation_random_sampler(self, sample_data):
        """Test that random sampler (shuffle=True) is rejected."""
        dataset = TensorDataset(sample_data)
        dl = DataLoader(dataset, batch_size=100, shuffle=True)

        # Should raise error about shuffle=True
        with pytest.raises(ValueError, match="shuffle=False"):
            pairwise_distances(dl, k=10, return_indices=True)

    def test_metadata_persists_across_calls(self, sample_data, dataloader):
        """Test that metadata persists and can be reused across multiple calls."""
        k = 10

        # First call caches metadata
        dist1, idx1 = pairwise_distances(dataloader, k=k, return_indices=True)

        # Get cached metadata
        metadata1 = get_dataloader_metadata(dataloader)
        assert metadata1 is not None
        assert metadata1["n_samples"] == len(sample_data)

        # Second call should still have metadata available
        dist2, idx2 = pairwise_distances(dataloader, k=k, return_indices=True)

        # Metadata should still be available after second call
        metadata2 = get_dataloader_metadata(dataloader)
        assert metadata2 is not None
        assert metadata2["n_samples"] == metadata1["n_samples"]
        assert metadata2["n_features"] == metadata1["n_features"]
        assert metadata2["dtype"] == metadata1["dtype"]

        # Results should be identical
        assert_close(dist1, dist2)
        assert torch.equal(idx1, idx2)

    def test_direct_function_caches_metadata(self, sample_data, dataloader):
        """Test that direct function also caches metadata."""
        k = 10

        # Call direct function
        dist, idx = pairwise_distances_faiss_from_dataloader(
            dataloader, k=k, metric="sqeuclidean"
        )

        # Verify metadata was cached
        metadata = get_dataloader_metadata(dataloader)
        assert metadata is not None
        assert metadata["n_samples"] == len(sample_data)
        assert metadata["n_features"] == sample_data.shape[1]


class TestIVFPQIndex:
    """Test IVFPQ index type for memory-efficient approximate search."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with dimension divisible by common M values."""
        torch.manual_seed(42)
        # Use 32 features (divisible by 8, 16, 32)
        n_samples = 2000
        n_features = 32
        return torch.randn(n_samples, n_features)

    @pytest.fixture
    def dataloader(self, sample_data):
        """Create DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=200, shuffle=False)

    def test_ivfpq_basic(self, sample_data, dataloader):
        """Test basic IVFPQ functionality."""
        k = 10
        config = FaissConfig(index_type="IVFPQ", nlist=50, nprobe=10, M=8, nbits=8)

        dist, idx = pairwise_distances(
            dataloader, k=k, backend=config, return_indices=True
        )

        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)
        # Indices should be valid
        assert (idx >= 0).all()
        assert (idx < len(sample_data)).all()

    def test_ivfpq_tensor_input(self, sample_data):
        """Test IVFPQ with tensor input."""
        k = 10
        config = FaissConfig(index_type="IVFPQ", nlist=50, nprobe=10, M=8, nbits=8)

        dist, idx = pairwise_distances(
            sample_data, k=k, backend=config, return_indices=True
        )

        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)

    def test_ivfpq_different_m_values(self, sample_data, dataloader):
        """Test IVFPQ with different M values."""
        k = 10
        # 32 is divisible by 8, 16, 32
        for M in [8, 16, 32]:
            config = FaissConfig(index_type="IVFPQ", nlist=50, nprobe=10, M=M, nbits=8)
            dist, idx = pairwise_distances(
                dataloader, k=k, backend=config, return_indices=True
            )
            assert dist.shape == (len(sample_data), k)

    def test_ivfpq_invalid_m(self):
        """Test that IVFPQ raises error when M doesn't divide dimension."""
        torch.manual_seed(42)
        # 33 features is not divisible by 8
        X = torch.randn(500, 33)
        config = FaissConfig(index_type="IVFPQ", nlist=50, nprobe=10, M=8, nbits=8)

        with pytest.raises(ValueError, match="must be divisible by M"):
            pairwise_distances(X, k=10, backend=config, return_indices=True)

    def test_ivfpq_vs_flat_recall(self, sample_data):
        """Test that IVFPQ has reasonable recall compared to exact search."""
        k = 10

        # Get exact results with Flat index
        dist_flat, idx_flat = pairwise_distances(
            sample_data, k=k, backend="faiss", return_indices=True
        )

        # Get approximate results with IVFPQ (high nprobe for better accuracy)
        config = FaissConfig(index_type="IVFPQ", nlist=50, nprobe=50, M=8, nbits=8)
        dist_pq, idx_pq = pairwise_distances(
            sample_data, k=k, backend=config, return_indices=True
        )

        # Compute recall: fraction of true neighbors found
        # With high nprobe, recall should be reasonable (>50% for k=10)
        recall = 0
        for i in range(len(sample_data)):
            true_neighbors = set(idx_flat[i].tolist())
            found_neighbors = set(idx_pq[i].tolist())
            recall += len(true_neighbors & found_neighbors) / k
        recall /= len(sample_data)

        # With nprobe=50 and nlist=50 (searching all clusters), recall should be high
        assert recall > 0.5, f"IVFPQ recall {recall:.2f} is too low"

    def test_ivfpq_config_repr(self):
        """Test that IVFPQ config has correct repr."""
        config = FaissConfig(index_type="IVFPQ", nlist=100, nprobe=10, M=16, nbits=8)
        repr_str = repr(config)

        assert "IVFPQ" in repr_str
        assert "M=16" in repr_str
        assert "nbits=8" in repr_str

    def test_ivfpq_exclude_diag(self, sample_data, dataloader):
        """Test IVFPQ with exclude_diag."""
        k = 10
        config = FaissConfig(index_type="IVFPQ", nlist=50, nprobe=10, M=8, nbits=8)

        dist, idx = pairwise_distances(
            dataloader, k=k, backend=config, exclude_diag=True, return_indices=True
        )

        assert dist.shape == (len(sample_data), k)
        assert idx.shape == (len(sample_data), k)
