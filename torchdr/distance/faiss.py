"""Distances based on Faiss backend."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
import warnings
from typing import Union, Optional, Dict, Any, Tuple, List

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    BatchSampler,
)

from torchdr.utils.faiss import faiss

LIST_METRICS_FAISS = ["euclidean", "sqeuclidean", "angular"]

# Cache for DataLoader metadata to avoid redundant iterations
_DATALOADER_METADATA_CACHE = {}


def get_dataloader_metadata(dataloader):
    """Get cached metadata for a DataLoader.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader to get metadata for.

    Returns
    -------
    metadata : dict or None
        Cached metadata dictionary with keys 'n_samples', 'n_features', 'dtype',
        or None if not cached.
    """
    return _DATALOADER_METADATA_CACHE.get(id(dataloader))


def _cache_dataloader_metadata(dataloader, metadata):
    """Cache metadata for a DataLoader.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader to cache metadata for.
    metadata : dict
        Metadata dictionary with keys 'n_samples', 'n_features', 'dtype'.
    """
    _DATALOADER_METADATA_CACHE[id(dataloader)] = metadata


def _is_deterministic_sampler(sampler):
    """Check if sampler provides deterministic iteration.

    Parameters
    ----------
    sampler : torch.utils.data.Sampler
        DataLoader sampler to check.

    Returns
    -------
    is_deterministic : bool
        True if sampler provides deterministic iteration order.
    """
    if isinstance(sampler, RandomSampler):
        return False

    if isinstance(sampler, SequentialSampler):
        return True

    if isinstance(sampler, BatchSampler):
        return _is_deterministic_sampler(sampler.sampler)

    if hasattr(sampler, "shuffle"):
        return not sampler.shuffle

    return True


def _validate_dataloader(dataloader):
    """Validate DataLoader is suitable for k-NN computation.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader to validate.

    Raises
    ------
    ValueError
        If DataLoader has shuffle=True or uses RandomSampler.
    """
    if not hasattr(dataloader, "sampler"):
        warnings.warn(
            "[TorchDR] Could not verify DataLoader has shuffle=False. "
            "Ensure deterministic iteration for correct k-NN results."
        )
        return

    if not _is_deterministic_sampler(dataloader.sampler):
        raise ValueError(
            "[TorchDR] DataLoader must have shuffle=False for deterministic "
            "iteration. Current sampler: {}. k-NN indices will be incorrect "
            "with shuffled data.".format(type(dataloader.sampler).__name__)
        )


class FaissConfig:
    """Configuration for FAISS k-NN computation.

    Parameters
    ----------
    temp_memory : Union[str, float], default='auto'
        GPU temporary memory allocation in GB.
        - 'auto': Use FAISS default temporary memory pool (typically a fixed size)
        - float/int: Explicit size in GB (e.g., 2.0 for 2GB)
        - 0: Disable pre-allocation (use cudaMalloc on demand)
        Only applies to GPU mode.
    device : int, default=0
        GPU device ID to use.
        Only applies when input is on CUDA.
    index_type : str, default='Flat'
        Type of FAISS index to use:
        - 'Flat': Exact brute-force search (slower but 100% accurate)
        - 'IVF': Inverted file index for approximate search (fast, ~95-99% accurate)
        - 'IVFPQ': Inverted file with Product Quantization for memory-efficient
          approximate search (very fast, ~90-95% accurate, highly compressed)
    nprobe : int, default=1
        Number of clusters to search in IVF indexes. Higher values increase
        accuracy but decrease speed. Only used with index_type='IVF' or 'IVFPQ'.
    nlist : int, default=100
        Number of clusters for IVF indexes. Typical values range from
        sqrt(n) to 4*sqrt(n) where n is the dataset size.
        Only used with index_type='IVF' or 'IVFPQ'.
    M : int, default=16
        Number of sub-quantizers for Product Quantization. The vector dimension
        must be divisible by M. Higher values give better accuracy but use more
        memory. Common values: 8, 16, 32, 64. Only used with index_type='IVFPQ'.
    nbits : int, default=8
        Number of bits per sub-quantizer code. Determines the number of centroids
        per subspace (2^nbits). Standard value is 8 (256 centroids per subspace).
        Only used with index_type='IVFPQ'.
    **kwargs
        Additional FAISS configuration options to pass to the underlying FAISS
        index config objects (e.g., for advanced memory management).
        Use at your own risk - some options may degrade result quality.

    Attributes
    ----------
    gpu_resources : Dict[int, Any]
        Dictionary mapping device IDs to their GPU resources (created on demand).

    Examples
    --------
    >>> # Basic configuration
    >>> config = FaissConfig()

    >>> # GPU configuration with specific device
    >>> config = FaissConfig(device=1)

    >>> # Custom memory allocation for large batch operations
    >>> config = FaissConfig(temp_memory=4.0)  # 4GB temp memory

    >>> # Memory-constrained environment
    >>> config = FaissConfig(temp_memory=0.5)  # 512MB only

    >>> # IVF approximate search for large datasets
    >>> config = FaissConfig(index_type="IVF", nlist=1000, nprobe=10)

    >>> # IVFPQ for very large datasets (100M+ vectors) with memory efficiency
    >>> config = FaissConfig(index_type="IVFPQ", nlist=4096, nprobe=64, M=16, nbits=8)

    Notes
    -----
    - Increasing temp_memory helps with large batch operations but reduces memory
      available for data storage
    - IVF indexes trade accuracy for speed and are recommended for datasets > 10M vectors
    - IVFPQ provides significant memory savings (e.g., 128D float32 vectors: 512 bytes
      -> ~32 bytes with M=16, nbits=8) at the cost of some accuracy
    - For IVFPQ, ensure the vector dimension is divisible by M
    """

    def __init__(
        self,
        temp_memory: Union[str, float] = "auto",
        device: int = 0,
        index_type: str = "Flat",
        nprobe: int = 1,
        nlist: int = 100,
        M: int = 16,
        nbits: int = 8,
        **kwargs,
    ):
        self.temp_memory = temp_memory
        self.device = device
        self.index_type = index_type
        self.nprobe = nprobe
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.faiss_kwargs = kwargs
        self.gpu_resources: Dict[int, Any] = {}

    def __repr__(self):
        parts = [
            f"temp_memory={self.temp_memory!r}",
            f"device={self.device}",
            f"index_type={self.index_type!r}",
            f"nprobe={self.nprobe}",
            f"nlist={self.nlist}",
        ]
        if self.index_type == "IVFPQ":
            parts.extend([f"M={self.M}", f"nbits={self.nbits}"])
        if self.faiss_kwargs:
            parts.append(f"**{self.faiss_kwargs}")
        return f"FaissConfig({', '.join(parts)})"


@torch.compiler.disable
def pairwise_distances_faiss(
    X: torch.Tensor,
    k: Union[int, torch.Tensor],
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    exclude_diag: bool = False,
    config: Optional[FaissConfig] = None,
    device: str = "auto",
):
    r"""Compute the k nearest neighbors using FAISS.

    Supported metrics are:
      - "euclidean": returns the Euclidean distance (square root of the squared distance)
      - "sqeuclidean": returns the squared Euclidean distance (as computed by FAISS)
      - "angular": returns the negative inner-product (after normalizing vectors)

    If Y is not provided then we assume a self–search and, if `exclude_diag` is True,
    the self–neighbor is removed from the results.

    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
        Query dataset.
    k : int or torch.Tensor
        Number of nearest neighbors to return. If tensor, will be converted to int.
        (If `exclude_diag` is True in a self–search, then k+1 neighbors are retrieved first.)
    Y : torch.Tensor of shape (m, d), optional
        Database dataset. If None, Y is set equal to X.
    metric : str, default "sqeuclidean"
        One of "euclidean", "sqeuclidean", or "angular".
    exclude_diag : bool, default False
        When True and Y is not provided (i.e. self–search), the self–neighbor (index i for query i)
        is excluded from the k results.
    config : FaissConfig, optional
        Configuration object for FAISS. If None, uses default settings.
        See FaissConfig documentation for available options.
    device : str, default="auto"
        Device to use for computation. If "auto", uses input device.
        If "cuda", uses FAISS GPU. If "cpu", uses FAISS CPU.
        Output remains on the specified device.

    Returns
    -------
    distances : torch.Tensor of shape (n, k)
        Nearest neighbor distances.
        For metric=="euclidean", distances are Euclidean (i.e. square root of L2^2).
        For metric=="sqeuclidean", distances are the squared Euclidean distances.
        For metric=="angular", distances are the (normalized) inner product scores.
    indices : torch.Tensor of shape (n, k)
        Indices of the k nearest neighbors.

    Examples
    --------
    >>> import torch
    >>> from torchdr.distance.faiss import pairwise_distances_faiss, FaissConfig

    >>> # Basic usage with default settings
    >>> X = torch.randn(1000, 128).cuda()
    >>> distances, indices = pairwise_distances_faiss(X, k=10)

    >>> # GPU configuration with specific device
    >>> config = FaissConfig(device=1)
    >>> distances, indices = pairwise_distances_faiss(X, k=10, config=config)

    >>> # Custom memory allocation for large batches
    >>> config = FaissConfig(temp_memory=4.0)  # 4GB temp memory
    >>> distances, indices = pairwise_distances_faiss(X, k=10, config=config)

    >>> # IVF approximate search for large datasets (100M vectors)
    >>> config = FaissConfig(index_type="IVF", nprobe=10)
    >>> distances, indices = pairwise_distances_faiss(X, k=10, config=config)
    """
    if metric not in LIST_METRICS_FAISS:
        raise ValueError(
            "[TorchDR] Only 'euclidean', 'sqeuclidean', and 'angular' metrics "
            "are supported for FAISS."
        )

    if config is None:
        config = FaissConfig()

    if isinstance(k, torch.Tensor):
        k = int(k.item())
    else:
        k = int(k)

    dtype = X.dtype
    X_np = X.detach().cpu().numpy().astype(np.float32)
    _, d = X_np.shape

    if Y is None or Y is X:
        Y_np = X_np
        do_exclude = exclude_diag
    else:
        Y_np = Y.detach().cpu().numpy().astype(np.float32)
        do_exclude = False

    if metric == "angular":
        flat_index = faiss.IndexFlatIP(d)
        metric_type = faiss.METRIC_INNER_PRODUCT
    elif metric in {"euclidean", "sqeuclidean"}:
        flat_index = faiss.IndexFlatL2(d)
        metric_type = faiss.METRIC_L2
    else:
        raise ValueError(f"[TorchDR] ERROR : Metric '{metric}' is not supported.")

    if config.index_type == "Flat":
        index = flat_index
    elif config.index_type == "IVF":
        n_vectors = len(Y_np)
        if config.nlist == 100 and n_vectors > 10000:
            config.nlist = min(int(4 * np.sqrt(n_vectors)), n_vectors // 40, 8192)
        index = faiss.IndexIVFFlat(flat_index, d, config.nlist, metric_type)
        index.nprobe = config.nprobe
    elif config.index_type == "IVFPQ":
        n_vectors = len(Y_np)
        if config.nlist == 100 and n_vectors > 10000:
            config.nlist = min(int(4 * np.sqrt(n_vectors)), n_vectors // 40, 8192)
        if d % config.M != 0:
            raise ValueError(
                f"[TorchDR] ERROR : Vector dimension {d} must be divisible by M={config.M} "
                f"for IVFPQ. Choose M from divisors of {d}."
            )
        index = faiss.IndexIVFPQ(flat_index, d, config.nlist, config.M, config.nbits)
        index.nprobe = config.nprobe
    else:
        raise ValueError(
            f"[TorchDR] ERROR : Index type '{config.index_type}' is not supported. "
            "Supported types are 'Flat', 'IVF', and 'IVFPQ'."
        )

    needs_training = config.index_type in ("IVF", "IVFPQ")

    if device == "auto":
        compute_device = X.device
    else:
        compute_device = torch.device(device)

    if compute_device.type == "cuda":
        if hasattr(faiss, "StandardGpuResources"):
            index = _setup_gpu_index(index, config, d)
        else:
            warnings.warn(
                "[TorchDR] WARNING: `faiss-gpu` not installed, using CPU for Faiss computations. "
                "This may be slow. For faster performance, install `faiss-gpu`."
            )

    if needs_training and not index.is_trained:
        train_data = Y_np
        max_train_points = 256 * config.nlist
        if len(Y_np) > max_train_points:
            sample_indices = np.random.choice(
                len(Y_np), max_train_points, replace=False
            )
            train_data = Y_np[sample_indices]
        index.train(train_data)

    index.add(Y_np)

    if do_exclude:
        k_search = k + 1
    else:
        k_search = k

    D, Ind = index.search(X_np, k_search)

    if metric == "euclidean":
        D = np.sqrt(D)
    elif metric == "angular":
        D = -D

    if do_exclude:
        D = D[:, 1:]
        Ind = Ind[:, 1:]

    distances = torch.from_numpy(D).to(compute_device).to(dtype)
    indices = torch.from_numpy(Ind).to(compute_device).long()

    return distances, indices


def _setup_gpu_index(index, config: FaissConfig, d: int):
    """Set up GPU index with configuration options.

    Parameters
    ----------
    index : faiss.Index
        CPU index to convert to GPU.
    config : FaissConfig
        Configuration object with GPU settings.
    d : int
        Dimension of the vectors.

    Returns
    -------
    gpu_index : faiss.GpuIndex
        Configured GPU index.
    """
    device_id = config.device if isinstance(config.device, int) else config.device[0]

    if device_id not in config.gpu_resources:
        res = faiss.StandardGpuResources()

        if config.temp_memory != "auto":
            temp_memory_bytes = int(config.temp_memory * 1024**3)
            res.setTempMemory(temp_memory_bytes)

        config.gpu_resources[device_id] = res
    else:
        res = config.gpu_resources[device_id]

    if isinstance(index, faiss.IndexFlatL2) or isinstance(index, faiss.IndexFlatIP):
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = device_id

        # Apply any additional kwargs
        for key, value in config.faiss_kwargs.items():
            if hasattr(flat_config, key):
                setattr(flat_config, key, value)

        if isinstance(index, faiss.IndexFlatL2):
            gpu_index = faiss.GpuIndexFlatL2(res, d, flat_config)
        else:
            gpu_index = faiss.GpuIndexFlatIP(res, d, flat_config)

    elif isinstance(index, faiss.IndexIVFPQ):
        # Handle IVFPQ index
        gpu_index = faiss.index_cpu_to_gpu(res, device_id, index)
        if hasattr(gpu_index, "nprobe"):
            gpu_index.nprobe = index.nprobe
    elif hasattr(index, "quantizer") and hasattr(index, "nprobe"):
        if hasattr(faiss, "GpuIndexIVFFlat"):
            ivf_config = faiss.GpuIndexIVFFlatConfig()
            ivf_config.device = device_id

            # Apply any additional kwargs
            for key, value in config.faiss_kwargs.items():
                if hasattr(ivf_config, key):
                    setattr(ivf_config, key, value)

            gpu_index = faiss.index_cpu_to_gpu(res, device_id, index)
            if hasattr(gpu_index, "nprobe"):
                gpu_index.nprobe = index.nprobe
        else:
            gpu_index = faiss.index_cpu_to_gpu(res, device_id, index)
    else:
        gpu_index = faiss.index_cpu_to_gpu(res, device_id, index)

    return gpu_index


@torch.compiler.disable
def pairwise_distances_faiss_from_dataloader(
    dataloader: DataLoader,
    k: int,
    metric: str = "sqeuclidean",
    exclude_diag: bool = False,
    config: Optional[FaissConfig] = None,
    device: str = "auto",
    distributed_ctx: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute k nearest neighbors using FAISS with DataLoader input.

    This function streams data from a DataLoader to build the FAISS index
    incrementally, avoiding the need to hold the full dataset in CPU RAM.
    Supports both single-GPU and multi-GPU (distributed) modes.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader yielding batches of data. Must be deterministic
        (shuffle=False) and yield tensors of shape (batch_size, n_features).
        In distributed mode, all ranks must iterate through the same data
        in the same order.
    k : int
        Number of nearest neighbors to return.
    metric : str, default "sqeuclidean"
        Distance metric. One of "euclidean", "sqeuclidean", or "angular".
    exclude_diag : bool, default False
        When True, exclude self-neighbors from results.
    config : FaissConfig, optional
        Configuration object for FAISS. If None, uses default settings.
    device : str, default "auto"
        Device for computation. If "auto", uses CUDA if available.
    distributed_ctx : DistributedContext, optional
        Distributed context for multi-GPU computation. When provided,
        each GPU computes k-NN for its assigned chunk of samples.

    Returns
    -------
    distances : torch.Tensor of shape (n_samples, k) or (chunk_size, k)
        k-NN distances. In distributed mode, returns only this rank's chunk.
    indices : torch.Tensor of shape (n_samples, k) or (chunk_size, k)
        k-NN indices. In distributed mode, returns only this rank's chunk.

    Examples
    --------
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> from torchdr.distance.faiss import pairwise_distances_faiss_from_dataloader
    >>> dataset = TensorDataset(torch.randn(10000, 128))
    >>> dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    >>> distances, indices = pairwise_distances_faiss_from_dataloader(
    ...     dataloader, k=15
    ... )

    Notes
    -----
    - DataLoader must have shuffle=False for deterministic iteration
    - In distributed mode, all ranks build the same full index
    - Memory efficient: only one batch in CPU RAM at a time
    - GPU memory still required for the full FAISS index
    """
    if metric not in LIST_METRICS_FAISS:
        raise ValueError(
            f"[TorchDR] Only {LIST_METRICS_FAISS} metrics are supported for FAISS."
        )

    # Validate DataLoader configuration
    _validate_dataloader(dataloader)

    if config is None:
        config = FaissConfig()

    # Determine compute device
    if distributed_ctx is not None and distributed_ctx.is_initialized:
        config = distributed_ctx.get_faiss_config(config)
        compute_device = torch.device(f"cuda:{distributed_ctx.local_rank}")
    elif device == "auto":
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(device)

    if not hasattr(dataloader.dataset, "__len__"):
        raise ValueError("[TorchDR] DataLoader dataset must have __len__ method.")

    # Build FAISS index and extract metadata in one pass
    index, metadata = _build_index_from_dataloader(
        dataloader, metric, config, compute_device
    )
    n_samples = metadata["n_samples"]
    dtype = metadata["dtype"]

    # Search for k-NN
    k_search = k + 1 if exclude_diag else k

    if distributed_ctx is not None and distributed_ctx.is_initialized:
        # Multi-GPU: each rank searches its chunk
        distances, indices = _search_chunk_from_dataloader(
            dataloader, index, k_search, distributed_ctx, n_samples, compute_device
        )
    else:
        # Single GPU: search all queries
        distances, indices = _search_all_from_dataloader(
            dataloader, index, k_search, compute_device
        )

    # Post-process results
    if metric == "euclidean":
        distances = torch.sqrt(distances)
    elif metric == "angular":
        distances = -distances

    if exclude_diag:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    return distances.to(dtype), indices


def _build_index_from_dataloader(
    dataloader: DataLoader,
    metric: str,
    config: FaissConfig,
    compute_device: torch.device,
):
    """Build FAISS index by streaming data from dataloader.

    Extracts metadata (n_samples, n_features, dtype) during the first pass,
    then builds the index. For Flat indices, only one pass through data is needed.
    For IVF indices, two passes are required (training + adding).

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader yielding batches of data.
    metric : str
        Distance metric.
    config : FaissConfig
        FAISS configuration.
    compute_device : torch.device
        Device for computation.

    Returns
    -------
    index : faiss.Index
        Built FAISS index containing all data.
    metadata : dict
        Dictionary with keys: 'n_samples', 'n_features', 'dtype'.
    """
    metadata = None
    index = None
    n_samples = len(dataloader.dataset)

    # For IVF/IVFPQ indices: first pass extracts metadata and trains
    if config.index_type in ("IVF", "IVFPQ"):
        collected = []
        total = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            # Extract metadata from first batch
            if metadata is None:
                metadata = {
                    "n_samples": n_samples,
                    "n_features": batch.shape[1],
                    "dtype": batch.dtype,
                }

                # Create index now that we know dimensions
                d = metadata["n_features"]
                if metric == "angular":
                    flat_index = faiss.IndexFlatIP(d)
                    metric_type = faiss.METRIC_INNER_PRODUCT
                else:
                    flat_index = faiss.IndexFlatL2(d)
                    metric_type = faiss.METRIC_L2

                nlist = config.nlist
                if nlist == 100 and n_samples > 10000:
                    nlist = min(int(4 * np.sqrt(n_samples)), n_samples // 40, 8192)

                if config.index_type == "IVFPQ":
                    if d % config.M != 0:
                        raise ValueError(
                            f"[TorchDR] ERROR : Vector dimension {d} must be divisible "
                            f"by M={config.M} for IVFPQ. Choose M from divisors of {d}."
                        )
                    index = faiss.IndexIVFPQ(
                        flat_index, d, nlist, config.M, config.nbits
                    )
                else:
                    index = faiss.IndexIVFFlat(flat_index, d, nlist, metric_type)
                index.nprobe = config.nprobe

                # Move to GPU if needed
                if compute_device.type == "cuda":
                    if hasattr(faiss, "StandardGpuResources"):
                        index = _setup_gpu_index(index, config, d)
                    else:
                        warnings.warn(
                            "[TorchDR] faiss-gpu not installed, using CPU. "
                            "Install faiss-gpu for faster computation."
                        )

            # Collect training data
            batch_np = batch.detach().cpu().numpy().astype(np.float32)
            # IVFPQ benefits from more training data for better codebooks
            if config.index_type == "IVFPQ":
                max_train = max(256 * index.nlist, 256 * config.M)
            else:
                max_train = 256 * index.nlist
            if total + len(batch_np) >= max_train:
                collected.append(batch_np[: max_train - total])
                break
            collected.append(batch_np)
            total += len(batch_np)

        # Train index
        train_data = np.vstack(collected)
        index.train(train_data)

    # For Flat indices: extract metadata from first batch only
    else:
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            metadata = {
                "n_samples": n_samples,
                "n_features": batch.shape[1],
                "dtype": batch.dtype,
            }

            # Create index
            d = metadata["n_features"]
            if metric == "angular":
                index = faiss.IndexFlatIP(d)
            else:
                index = faiss.IndexFlatL2(d)

            # Move to GPU if needed
            if compute_device.type == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    index = _setup_gpu_index(index, config, d)
                else:
                    warnings.warn(
                        "[TorchDR] faiss-gpu not installed, using CPU. "
                        "Install faiss-gpu for faster computation."
                    )
            break

    # Second pass: add all data to index
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch_np = batch.detach().cpu().numpy().astype(np.float32)
        index.add(batch_np)

    # Cache metadata for later reuse
    _cache_dataloader_metadata(dataloader, metadata)

    return index, metadata


def _search_all_from_dataloader(
    dataloader: DataLoader,
    index,
    k: int,
    compute_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Search k-NN for all samples in dataloader (single-GPU mode).

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader with query samples.
    index : faiss.Index
        FAISS index to search.
    k : int
        Number of neighbors to find.
    compute_device : torch.device
        Device for output tensors.

    Returns
    -------
    distances : torch.Tensor of shape (n_samples, k)
        k-NN distances.
    indices : torch.Tensor of shape (n_samples, k)
        k-NN indices.
    """
    distances_list: List[torch.Tensor] = []
    indices_list: List[torch.Tensor] = []

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch_np = batch.detach().cpu().numpy().astype(np.float32)

        D, Ind = index.search(batch_np, k)

        distances_list.append(torch.from_numpy(D))
        indices_list.append(torch.from_numpy(Ind))

    distances = torch.cat(distances_list, dim=0).to(compute_device)
    indices = torch.cat(indices_list, dim=0).to(compute_device).long()

    return distances, indices


def _search_chunk_from_dataloader(
    dataloader: DataLoader,
    index,
    k: int,
    distributed_ctx,
    n_samples: int,
    compute_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Search k-NN for this rank's chunk of samples (distributed mode).

    Each rank iterates through the dataloader but only processes
    the samples assigned to its chunk.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader with all samples.
    index : faiss.Index
        FAISS index to search (contains full dataset).
    k : int
        Number of neighbors to find.
    distributed_ctx : DistributedContext
        Distributed context with rank info.
    n_samples : int
        Total number of samples in dataset.
    compute_device : torch.device
        Device for output tensors.

    Returns
    -------
    distances : torch.Tensor of shape (chunk_size, k)
        k-NN distances for this rank's chunk.
    indices : torch.Tensor of shape (chunk_size, k)
        k-NN indices for this rank's chunk.
    """
    chunk_start, chunk_end = distributed_ctx.compute_chunk_bounds(n_samples)

    distances_list: List[torch.Tensor] = []
    indices_list: List[torch.Tensor] = []

    current_idx = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        batch_size = len(batch)
        batch_end = current_idx + batch_size

        # Check if this batch overlaps with our chunk
        if batch_end > chunk_start and current_idx < chunk_end:
            # Compute overlap indices within this batch
            start_in_batch = max(0, chunk_start - current_idx)
            end_in_batch = min(batch_size, chunk_end - current_idx)

            # Extract our portion of this batch
            my_batch = batch[start_in_batch:end_in_batch]
            batch_np = my_batch.detach().cpu().numpy().astype(np.float32)

            # Search
            D, Ind = index.search(batch_np, k)

            distances_list.append(torch.from_numpy(D))
            indices_list.append(torch.from_numpy(Ind))

        current_idx = batch_end

        # Early exit if we've processed our entire chunk
        if current_idx >= chunk_end:
            break

    # Handle empty chunk case (when n_samples < world_size)
    if len(distances_list) == 0:
        return (
            torch.empty(0, k, device=compute_device),
            torch.empty(0, k, dtype=torch.long, device=compute_device),
        )

    distances = torch.cat(distances_list, dim=0).to(compute_device)
    indices = torch.cat(indices_list, dim=0).to(compute_device).long()

    return distances, indices
