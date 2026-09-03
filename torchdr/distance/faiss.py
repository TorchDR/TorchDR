"""Distances based on Faiss backend."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import torch.distributed as dist
import numpy as np
import warnings
from weakref import WeakKeyDictionary
from typing import Union, Optional, Dict, Any, Callable, Iterable, Tuple, List

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    BatchSampler,
)

from torchdr.distributed.input_contract import collective_device
from torchdr.utils.faiss import faiss, faiss_torch_interop
from torchdr.utils.faiss_runtime import (
    faiss_device_scope,
    faiss_gpu_available,
    get_gpu_resources,
)

LIST_METRICS_FAISS = ["euclidean", "sqeuclidean", "angular"]

# Rows per FAISS add/search call when a DataLoader does not dictate one.
_AUTO_STREAM_ROWS = 65536

# Cache for DataLoader metadata to avoid redundant iterations
_DATALOADER_METADATA_CACHE = WeakKeyDictionary()


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
    return _DATALOADER_METADATA_CACHE.get(dataloader)


def _cache_dataloader_metadata(dataloader, metadata):
    """Cache metadata for a DataLoader.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader to cache metadata for.
    metadata : dict
        Metadata dictionary with keys 'n_samples', 'n_features', 'dtype'.
    """
    _DATALOADER_METADATA_CACHE[dataloader] = metadata


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
        Only applies to GPU mode. The pool belongs to the calling thread rather
        than to this config: searches on the same thread and device reuse it.
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
    stream_batch_size : Union[str, int], default='auto'
        Rows handed to FAISS per add or search call when streaming from a
        DataLoader. 'auto' regroups batches for a GPU index, merging small ones
        and splitting large ones, so the batch size chosen for the training loop
        does not also decide the size of a FAISS call, and leaves a CPU index to
        consume batches as they arrive. An integer overrides the target. FAISS
        picks its exact-search kernel from the number of queries in a call, so
        a different value can move a distance by the float32 cancellation
        residual without changing the neighbors it ranks.
    **kwargs
        Additional FAISS configuration options to pass to the underlying FAISS
        index config objects (e.g., for advanced memory management).
        Use at your own risk - some options may degrade result quality.

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
      available for data storage. FAISS defaults to a pool of roughly 1.5GB per
      device, which the first GPU search allocates and the calling thread keeps.
      Set temp_memory explicitly to bound it.
    - IVF indexes trade accuracy for speed and are recommended for datasets > 10M vectors
    - IVFPQ provides significant memory savings (e.g., 128D float32 vectors: 512 bytes
      -> ~32 bytes with M=16, nbits=8) at the cost of some accuracy
    - For IVFPQ, ensure the vector dimension is divisible by M
    - The configuration is plain data. FAISS GPU resources are owned by the
      calling thread, not by the configuration, so a config stays copyable and
      picklable after it has been used on a GPU. Resources are not shared across
      threads because FAISS does not make ``StandardGpuResources`` thread-safe.
      Consequently, concurrent CPU threads targeting the same GPU each keep a
      separate temporary pool.
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
        stream_batch_size: Union[str, int] = "auto",
        **kwargs,
    ):
        self.temp_memory = temp_memory
        self.device = device
        self.index_type = index_type
        self.nprobe = nprobe
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.stream_batch_size = stream_batch_size
        self.faiss_kwargs = kwargs

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
        if self.stream_batch_size != "auto":
            parts.append(f"stream_batch_size={self.stream_batch_size}")
        if self.faiss_kwargs:
            parts.append(f"**{self.faiss_kwargs}")
        return f"FaissConfig({', '.join(parts)})"


def _index_device_id(config: FaissConfig) -> int:
    """CUDA ordinal the FAISS index runs on. A sequence selects its first device."""
    device = config.device
    return int(device) if isinstance(device, int) else int(device[0])


def remove_self_neighbors(
    distances: torch.Tensor,
    indices: torch.Tensor,
    query_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Drop each query's own index from a ``k + 1`` neighbor search.

    A self-search retrieves ``k + 1`` neighbors so that ``k`` remain once the
    query itself is removed. The query is not reliably the first result: exact
    ties put a duplicate observation ahead of it, the ``angular`` metric ranks
    by raw inner product so a longer vector can outrank it, and an approximate
    index may not return it at all. Selecting by index is therefore used rather
    than dropping column 0.

    When a row does not contain its query id, all ``k + 1`` results are valid
    neighbors and the farthest one is dropped instead.

    Parameters
    ----------
    distances : torch.Tensor of shape (n, k + 1)
        Neighbor distances, sorted by increasing distance.
    indices : torch.Tensor of shape (n, k + 1)
        Neighbor indices in the database, aligned with ``distances``.
    query_ids : torch.Tensor of shape (n,)
        Database index of each query row. In distributed mode this is the
        global index of the row, not its position within the local chunk.

    Returns
    -------
    distances : torch.Tensor of shape (n, k)
        Neighbor distances without the self-neighbor.
    indices : torch.Tensor of shape (n, k)
        Neighbor indices without the self-neighbor.
    """
    k_plus_one = indices.shape[1]
    query_ids = query_ids.to(device=indices.device)

    # An exact search over distinct points puts every query first. Slicing that
    # off returns views and allocates nothing, so the common case keeps the cost
    # of one comparison and pays no memory for the general case below.
    if bool((indices[:, 0] == query_ids).all()):
        return distances[:, 1:], indices[:, 1:]

    is_self = indices == query_ids.unsqueeze(1)
    # argmax gives the first match, or 0 for a row with none; such a row has
    # k + 1 valid neighbors, so its last (farthest) column is dropped instead.
    dropped = torch.where(
        is_self.any(dim=1),
        is_self.to(torch.uint8).argmax(dim=1),
        k_plus_one - 1,
    )

    # Output column j takes input column j before the dropped position and
    # column j + 1 after it. Both operands are views, so the only tensors
    # materialized are the mask and the two outputs.
    after = torch.arange(k_plus_one - 1, device=indices.device) >= dropped.unsqueeze(1)

    return (
        torch.where(after, distances[:, 1:], distances[:, :-1]),
        torch.where(after, indices[:, 1:], indices[:, :-1]),
    )


@torch.compiler.disable
def pairwise_distances_faiss(
    X: torch.Tensor,
    k: Union[int, torch.Tensor],
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    exclude_diag: bool = False,
    config: Optional[FaissConfig] = None,
    device: str = "auto",
    query_ids: Optional[torch.Tensor] = None,
    distributed_ctx: Optional[Any] = None,
):
    r"""Compute the k nearest neighbors using FAISS.

    Supported metrics are:
      - "euclidean": returns the Euclidean distance (square root of the squared distance)
      - "sqeuclidean": returns the squared Euclidean distance (as computed by FAISS)
      - "angular": returns the negative inner product. Inputs are not normalized;
        normalize them beforehand to obtain cosine-similarity neighbor ordering.

    If Y is not provided then we assume a self–search and, if `exclude_diag` is True,
    the self–neighbor is removed from the results. When X is a chunk of a larger
    database, pass `query_ids` so that `exclude_diag` still applies.

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
        When True, the self–neighbor (the query's own database row) is excluded
        from the k results. Requires a self–search, unless `query_ids` is given.
        The removal matches on the index, so it is correct when duplicate
        observations, the ``angular`` metric or an approximate index place the
        query away from the first position.
    config : FaissConfig, optional
        Configuration object for FAISS. If None, uses default settings.
        See FaissConfig documentation for available options.
    device : str, default="auto"
        Device to use for computation. If "auto", uses input device.
        If "cuda", uses FAISS GPU. If "cpu", uses FAISS CPU.
        Output remains on the specified device.
    query_ids : torch.Tensor of shape (n,), optional
        Row of Y that each query of X corresponds to. Only used when
        `exclude_diag` is True, and only needed when X is a chunk of Y rather
        than Y itself, as in distributed search. Defaults to `arange(n)`.
    distributed_ctx : DistributedContext, optional
        Context of the ranks that share this search. Every rank indexes the
        same database, so an approximate index is trained once on rank 0 and
        broadcast rather than trained again on each of them.

    Returns
    -------
    distances : torch.Tensor of shape (n, k)
        Nearest neighbor distances.
        For metric=="euclidean", distances are Euclidean (i.e. square root of L2^2).
        For metric=="sqeuclidean", distances are the squared Euclidean distances.
        For metric=="angular", distances are the negative raw inner-product scores.
    indices : torch.Tensor of shape (n, k)
        Indices of the k nearest neighbors.

    Notes
    -----
    FAISS computes index distances in float32. Tensor inputs are passed directly
    to FAISS on CPU or GPU when the installed FAISS build provides its PyTorch
    interoperability wrappers; results are cast back to the input dtype.

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
    _, d = X.shape

    if Y is None or Y is X:
        Y = X
        do_exclude = exclude_diag
    else:
        # A chunk of the database can still drop its self-neighbors, but only
        # if the caller states which database row each query came from.
        do_exclude = exclude_diag and query_ids is not None

    if X.dtype != torch.float32 or Y.dtype != torch.float32:
        warnings.warn(
            "[TorchDR] FAISS computes distances in float32; input values will "
            "be converted and results cast back to the input dtype.",
            UserWarning,
            stacklevel=2,
        )

    if device == "auto":
        compute_device = X.device
    else:
        compute_device = torch.device(device)

    index_device, use_gpu_index = _index_input_device(config, compute_device)
    if compute_device.type == "cuda" and not use_gpu_index:
        warnings.warn(
            "[TorchDR] WARNING: `faiss-gpu` not installed, using CPU for Faiss computations. "
            "This may be slow. For faster performance, install `faiss-gpu`."
        )

    # Device the index lives on, or None when FAISS stays on the host.
    faiss_device = (
        torch.device("cuda", _index_device_id(config)) if use_gpu_index else None
    )

    index = _create_index(metric, config, d, len(Y), use_gpu_index)

    X_faiss = X.detach().to(device=index_device, dtype=torch.float32).contiguous()
    Y_faiss = Y.detach().to(device=index_device, dtype=torch.float32).contiguous()

    if not faiss_torch_interop:
        X_faiss = X_faiss.cpu().numpy()
        Y_faiss = Y_faiss.cpu().numpy()

    if do_exclude:
        k_search = k + 1
    else:
        k_search = k

    # Every FAISS call runs with the index device current, which is what makes
    # FAISS adopt the PyTorch stream holding the writes to X_faiss and Y_faiss.
    with faiss_device_scope(faiss_device):
        if not index.is_trained:
            index = _train_index(
                index,
                lambda: _training_sample(Y_faiss, 256 * index.nlist),
                config,
                d,
                use_gpu_index,
                distributed_ctx,
            )

        index.add(Y_faiss)

        D, Ind = index.search(X_faiss, k_search)

    if metric == "euclidean":
        if isinstance(D, torch.Tensor):
            D = torch.sqrt(D)
        else:
            D = np.sqrt(D)
    elif metric == "angular":
        D = -D

    if not isinstance(D, torch.Tensor):
        D = torch.from_numpy(D)
        Ind = torch.from_numpy(Ind)

    # Drop the self-neighbor before casting, so that only the (n, k) result is
    # materialized in the output dtype rather than the wider (n, k + 1) search.
    if do_exclude:
        if query_ids is None:
            query_ids = torch.arange(Ind.shape[0], device=Ind.device)
        D, Ind = remove_self_neighbors(D, Ind, query_ids)

    distances = D.to(device=compute_device, dtype=dtype)
    indices = Ind.to(device=compute_device, dtype=torch.long)

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

    Notes
    -----
    The resource is owned by the calling thread rather than by ``config``, so
    repeated calls on the same thread and device reuse one temporary memory pool
    while a config stays free of live FAISS objects.
    """
    device_id = _index_device_id(config)
    res = get_gpu_resources(device_id, config.temp_memory)

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
        When True, exclude self-neighbors from results. Rows are matched by
        their global index, which stays correct in distributed mode where a
        rank only queries its own chunk.
    config : FaissConfig, optional
        Configuration object for FAISS. If None, uses default settings.
    device : str, default "auto"
        Device the index is built on. "auto" uses CUDA when available, so host
        batches are streamed to a GPU index, and returns the results on the
        device the batches came from, which is what the tensor path does. An
        explicit device sets both the index and the output device.
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
    - Batches reach FAISS as tensors where the build ships the PyTorch
      wrappers, avoiding a NumPy round trip. Consecutive batches are regrouped
      into FAISS calls of ``config.stream_batch_size`` rows.
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
    distributed = distributed_ctx is not None and distributed_ctx.is_initialized
    if distributed:
        config = distributed_ctx.get_faiss_config(config)
        compute_device = torch.device(f"cuda:{distributed_ctx.local_rank}")
    elif device == "auto":
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(device)

    if not hasattr(dataloader.dataset, "__len__"):
        raise ValueError("[TorchDR] DataLoader dataset must have __len__ method.")

    index_device, use_gpu_index = _index_input_device(config, compute_device)
    if compute_device.type == "cuda" and not use_gpu_index:
        warnings.warn(
            "[TorchDR] faiss-gpu not installed, using CPU. "
            "Install faiss-gpu for faster computation."
        )

    def stage(batch):
        return _stage_batch(batch, index_device, to_numpy=not faiss_torch_interop)

    group_rows = _resolve_stream_rows(config, use_gpu_index)
    faiss_device = (
        torch.device("cuda", _index_device_id(config)) if use_gpu_index else None
    )

    # Build FAISS index and extract metadata in one pass
    with faiss_device_scope(faiss_device):
        index, metadata = _build_index_from_dataloader(
            dataloader,
            metric,
            config,
            stage,
            group_rows,
            use_gpu_index,
            distributed_ctx if distributed else None,
        )
    n_samples = metadata["n_samples"]
    dtype = metadata["dtype"]

    # Results follow the batches, the way the tensor path follows ``X.device``.
    # An explicit device, and distributed mode, pin the output instead.
    if device == "auto" and not distributed:
        output_device = metadata["device"]
    else:
        output_device = compute_device

    # Search for k-NN
    k_search = k + 1 if exclude_diag else k

    if distributed:
        # Multi-GPU: each rank searches its chunk
        chunk_start, chunk_end = distributed_ctx.compute_chunk_bounds(n_samples)
        queries = _chunk_batches(dataloader, chunk_start, chunk_end)
    else:
        # Single GPU: search all queries
        chunk_start, chunk_end = 0, n_samples
        queries = _batches(dataloader)

    with faiss_device_scope(faiss_device):
        distances, indices = _search_from_dataloader(
            queries, index, k_search, stage, group_rows, output_device
        )

    # Post-process results
    if metric == "euclidean":
        distances = torch.sqrt(distances)
    elif metric == "angular":
        distances = -distances

    if exclude_diag:
        # Global row index of each query, so that the self-neighbor is removed
        # by identity rather than by position.
        query_ids = torch.arange(chunk_start, chunk_end, device=indices.device)
        distances, indices = remove_self_neighbors(distances, indices, query_ids)

    return distances.to(dtype), indices


def _batch_tensor(batch) -> torch.Tensor:
    """Data tensor of a DataLoader batch, which datasets often wrap in a tuple."""
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    return batch


def _batch_metadata(n_samples: int, batch: torch.Tensor) -> Dict[str, Any]:
    """Describe a DataLoader from its first batch."""
    return {
        "n_samples": n_samples,
        "n_features": batch.shape[1],
        "dtype": batch.dtype,
        "device": batch.device,
    }


def _stage_batch(batch: torch.Tensor, device: torch.device, to_numpy: bool = False):
    """Present one DataLoader batch to FAISS without a NumPy round trip.

    Batches are made contiguous float32 and moved to the device the index reads
    from. DataLoader-managed pinned batches retain their asynchronous upload;
    ordinary host batches use PyTorch's direct device transfer.

    Parameters
    ----------
    device : torch.device
        Device FAISS reads its inputs from. CPU for a CPU index.
    to_numpy : bool, default=False
        Stage through NumPy instead, for FAISS builds that ship without the
        PyTorch wrappers.
    """

    batch = batch.detach().to(dtype=torch.float32).contiguous()

    if to_numpy:
        return batch.cpu().numpy()
    if device.type != "cuda":
        return batch.cpu()
    return batch.to(device, non_blocking=batch.is_pinned())


def _concatenate(chunks: List[Any]):
    """Join staged chunks, whichever staging format they use."""
    if len(chunks) == 1:
        return chunks[0]
    if isinstance(chunks[0], torch.Tensor):
        return torch.cat(chunks, dim=0)
    return np.vstack(chunks)


def _batches(dataloader: DataLoader):
    """Yield the data tensor of every batch of a DataLoader."""
    for batch in dataloader:
        yield _batch_tensor(batch)


def _stream(
    batches: Iterable[torch.Tensor],
    stage: Callable[[torch.Tensor], Any],
    group_rows: Optional[int],
):
    """Stage batches for FAISS, in groups of about ``group_rows`` rows.

    The DataLoader's batch size is usually chosen for the training loop, not
    for FAISS. Grouping decouples the two: small batches are merged so each
    call has enough work to amortize its overhead, and large ones are split so
    a single call cannot blow up FAISS's temporary memory. Only whole batches
    are ever held at once, so the dataset still never has to fit in memory.
    ``None`` hands every batch over as it arrives.
    """
    if group_rows is None:
        for batch in batches:
            yield stage(batch)
        return

    pending: List[Any] = []
    rows = 0

    for batch in batches:
        staged = stage(batch)

        if len(staged) >= group_rows:
            if pending:
                yield _concatenate(pending)
                pending, rows = [], 0
            for start in range(0, len(staged), group_rows):
                yield staged[start : start + group_rows]
            continue

        pending.append(staged)
        rows += len(staged)
        if rows >= group_rows:
            yield _concatenate(pending)
            pending, rows = [], 0

    if pending:
        yield _concatenate(pending)


def _as_tensor(result) -> torch.Tensor:
    """FAISS returns tensors with the PyTorch wrappers and arrays without."""
    return result if isinstance(result, torch.Tensor) else torch.from_numpy(result)


def _resolve_stream_rows(config: FaissConfig, use_gpu_index: bool) -> Optional[int]:
    """Rows per FAISS call when streaming a DataLoader, or None to pass through."""
    if config.stream_batch_size != "auto":
        rows = int(config.stream_batch_size)
        if rows < 1:
            raise ValueError(
                "[TorchDR] ERROR : stream_batch_size must be a positive number "
                f"of rows or 'auto', got {config.stream_batch_size!r}."
            )
        return rows
    # A CPU index gains nothing from regrouping and would pay for the copy, so
    # only a GPU index, whose calls carry a transfer and a launch, gets one.
    return _AUTO_STREAM_ROWS if use_gpu_index else None


def _index_input_device(config: FaissConfig, compute_device: torch.device):
    """Device FAISS reads from, and whether the index itself lives on the GPU."""
    use_gpu_index = compute_device.type == "cuda" and faiss_gpu_available()
    if use_gpu_index and faiss_torch_interop:
        return torch.device("cuda", _index_device_id(config)), use_gpu_index
    return torch.device("cpu"), use_gpu_index


def _create_index(
    metric: str, config: FaissConfig, d: int, n_samples: int, use_gpu_index: bool
):
    """Create the empty index that the training and adding passes will fill.

    An automatic ``nlist`` is resolved here rather than written back into
    ``config``, so the configuration the caller passed in keeps its own values
    and describes the same search whichever dataset it is used on.
    """
    if config.index_type not in ("Flat", "IVF", "IVFPQ"):
        raise ValueError(
            f"[TorchDR] ERROR : Index type '{config.index_type}' is not supported. "
            "Supported types are 'Flat', 'IVF', and 'IVFPQ'."
        )

    if metric == "angular":
        flat_index = faiss.IndexFlatIP(d)
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        flat_index = faiss.IndexFlatL2(d)
        metric_type = faiss.METRIC_L2

    if config.index_type in ("IVF", "IVFPQ"):
        nlist = config.nlist
        if nlist == 100 and n_samples > 10000:
            nlist = min(int(4 * np.sqrt(n_samples)), n_samples // 40, 8192)

        if config.index_type == "IVFPQ":
            if d % config.M != 0:
                raise ValueError(
                    f"[TorchDR] ERROR : Vector dimension {d} must be divisible "
                    f"by M={config.M} for IVFPQ. Choose M from divisors of {d}."
                )
            index = faiss.IndexIVFPQ(flat_index, d, nlist, config.M, config.nbits)
        else:
            index = faiss.IndexIVFFlat(flat_index, d, nlist, metric_type)
        index.nprobe = config.nprobe
    else:
        index = flat_index

    if use_gpu_index:
        index = _setup_gpu_index(index, config, d)
    return index


def _training_sample(data, max_train_points: int):
    """Rows an IVF/PQ index trains on: at most ``max_train_points`` of ``data``.

    The draw comes from the global NumPy generator, which ``random_state``
    seeds, so a seeded estimator trains on the same rows from one run to the
    next.
    """
    if len(data) <= max_train_points:
        return data

    sample_indices = np.random.choice(len(data), max_train_points, replace=False)
    if isinstance(data, torch.Tensor):
        sample_indices = torch.as_tensor(sample_indices, device=data.device)
    return data[sample_indices]


def _shares_training(distributed_ctx) -> bool:
    """Whether one rank trains for the group instead of each rank training."""
    return (
        distributed_ctx is not None
        and distributed_ctx.is_initialized
        and distributed_ctx.world_size > 1
        and dist.is_initialized()
    )


def _train_index(
    index,
    training_rows: Callable[[], Any],
    config: FaissConfig,
    d: int,
    use_gpu_index: bool,
    distributed_ctx,
):
    """Train an IVF/PQ index once for the group rather than once per rank.

    Every rank indexes the same database, so training on all of them repeats
    identical work, and because each draws its own training sample the ranks
    end up with different quantizers and disagree on the neighbors of a query.
    Rank 0 trains and broadcasts the trained, still empty, index; the others
    receive it and go straight to adding their vectors.

    Parameters
    ----------
    index : faiss.Index
        Untrained index, already on its final device.
    training_rows : callable
        Produces the rows to train on. It is only called on the rank that
        trains, so the others never materialize a training sample.
    config : FaissConfig
        Configuration used to move a received index back onto the GPU.
    d : int
        Dimension of the vectors.
    use_gpu_index : bool
        Whether the index lives on the GPU.
    distributed_ctx : DistributedContext or None
        Context of the ranks sharing the index. None trains locally.

    Returns
    -------
    index : faiss.Index
        The trained index, which on a receiving rank is a new object.
    """
    if not _shares_training(distributed_ctx):
        index.train(training_rows())
        return index

    trainer = 0
    payload = None
    trained = True
    if distributed_ctx.rank == trainer:
        try:
            index.train(training_rows())
            template = faiss.index_gpu_to_cpu(index) if use_gpu_index else index
            payload = faiss.serialize_index(template)
        except Exception as error:  # reported on every rank, not just this one
            trained = False
            reason = f"{type(error).__name__}: {error}"
            payload = np.frombuffer(reason.encode(), dtype=np.uint8).copy()

    # The header travels first: the receivers need the length to size their
    # buffer, and it carries a failure on the same two collectives, so a rank
    # that could not train never leaves the others waiting on a third.
    device = collective_device(distributed_ctx)
    header = torch.zeros(2, dtype=torch.int64, device=device)
    if payload is not None:
        header[0] = trained
        header[1] = payload.size
    dist.broadcast(header, src=trainer)

    buffer = torch.empty(int(header[1]), dtype=torch.uint8, device=device)
    if payload is not None:
        buffer.copy_(torch.from_numpy(payload))
    dist.broadcast(buffer, src=trainer)

    if not int(header[0]):
        raise RuntimeError(
            f"[TorchDR] ERROR : rank {trainer} failed to train the "
            f"'{config.index_type}' index that every rank shares: "
            + bytes(buffer.cpu().numpy()).decode("utf-8", "replace")
        )

    if distributed_ctx.rank == trainer:
        return index

    index = faiss.deserialize_index(buffer.cpu().numpy())
    return _setup_gpu_index(index, config, d) if use_gpu_index else index


def _reserve_index_capacity(index, n_vectors: int) -> None:
    """Pre-allocate storage for the incremental additions that follow.

    A GPU index grows by reallocating and copying its whole database, so a
    stream of small additions moves the same vectors again and again. Only some
    index types expose a reservation call, hence the feature detection; when it
    is missing or the build rejects the request, the additions simply grow the
    index as before.
    """
    reserve = getattr(index, "reserveVecs", None) or getattr(
        index, "reserveMemory", None
    )
    if reserve is None:
        return
    try:
        reserve(n_vectors)
    except (RuntimeError, TypeError):  # pragma: no cover - build specific
        pass


def _build_index_from_dataloader(
    dataloader: DataLoader,
    metric: str,
    config: FaissConfig,
    stage: Callable[[torch.Tensor], Any],
    group_rows: Optional[int],
    use_gpu_index: bool,
    distributed_ctx=None,
):
    """Build FAISS index by streaming data from dataloader.

    Extracts metadata (n_samples, n_features, dtype, device) from the first
    batch, then builds the index. For Flat indices, only one pass through data
    is needed. For IVF indices, two passes are required (training + adding).

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader yielding batches of data.
    metric : str
        Distance metric.
    config : FaissConfig
        FAISS configuration.
    stage : callable
        Converts a batch into something FAISS can consume directly.
    group_rows : int or None
        Rows per add call, or None to add each batch as it arrives.
    use_gpu_index : bool
        Whether the index is moved to the GPU.
    distributed_ctx : DistributedContext, optional
        Context of the ranks sharing the index. Only rank 0 reads training
        rows; the others receive the trained index from it.

    Returns
    -------
    index : faiss.Index
        Built FAISS index containing all data.
    metadata : dict
        Dictionary with keys: 'n_samples', 'n_features', 'dtype', 'device'.
    """
    metadata = None
    index = None
    n_samples = len(dataloader.dataset)

    # First pass: describe the stream from its first batch, then collect
    # training rows if the index needs them. A Flat index stops right away, and
    # so does a rank that will be given its trained index by rank 0.
    collected: List[Any] = []
    total = 0

    for batch in dataloader:
        batch = _batch_tensor(batch)

        if metadata is None:
            metadata = _batch_metadata(n_samples, batch)
            index = _create_index(
                metric, config, metadata["n_features"], n_samples, use_gpu_index
            )
            if index.is_trained:
                break
            if _shares_training(distributed_ctx) and distributed_ctx.rank != 0:
                break
            # IVFPQ benefits from more training data for better codebooks
            max_train = 256 * index.nlist
            if config.index_type == "IVFPQ":
                max_train = max(max_train, 256 * config.M)

        staged = stage(batch)
        if total + len(staged) >= max_train:
            collected.append(staged[: max_train - total])
            break
        collected.append(staged)
        total += len(staged)

    if not index.is_trained:
        index = _train_index(
            index,
            lambda: _concatenate(collected),
            config,
            metadata["n_features"],
            use_gpu_index,
            distributed_ctx,
        )

    # Second pass: add all data to index
    _reserve_index_capacity(index, n_samples)
    for group in _stream(_batches(dataloader), stage, group_rows):
        index.add(group)

    # Cache metadata for later reuse
    _cache_dataloader_metadata(dataloader, metadata)

    return index, metadata


def _search_from_dataloader(
    batches: Iterable[torch.Tensor],
    index,
    k: int,
    stage: Callable[[torch.Tensor], Any],
    group_rows: Optional[int],
    output_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Search k-NN for a stream of query batches.

    Parameters
    ----------
    batches : iterable of torch.Tensor
        Query batches, in dataset order. Either every batch of the DataLoader
        or, in distributed mode, this rank's slice of them.
    index : faiss.Index
        FAISS index to search.
    k : int
        Number of neighbors to find.
    stage : callable
        Converts a batch into something FAISS can consume directly.
    group_rows : int or None
        Rows per search call, or None to search each batch as it arrives.
    output_device : torch.device
        Device for output tensors. Results are moved group by group, so a host
        output never accumulates on the device.

    Returns
    -------
    distances : torch.Tensor of shape (n_queries, k)
        k-NN distances.
    indices : torch.Tensor of shape (n_queries, k)
        k-NN indices.
    """
    distances_list: List[torch.Tensor] = []
    indices_list: List[torch.Tensor] = []

    for group in _stream(batches, stage, group_rows):
        D, Ind = index.search(group, k)

        distances_list.append(_as_tensor(D).to(output_device))
        indices_list.append(_as_tensor(Ind).to(output_device))

    # Empty chunk case, when n_samples < world_size.
    if not distances_list:
        return (
            torch.empty(0, k, device=output_device),
            torch.empty(0, k, dtype=torch.long, device=output_device),
        )

    distances = torch.cat(distances_list, dim=0)
    indices = torch.cat(indices_list, dim=0).long()

    return distances, indices


def _chunk_batches(dataloader: DataLoader, start: int, end: int):
    """Yield the rows of each batch that fall in ``[start, end)``.

    Every rank walks the same DataLoader, so a rank's chunk is the slice of the
    stream it is responsible for querying. Iteration stops once the chunk is
    exhausted.
    """
    offset = 0
    for batch in dataloader:
        batch = _batch_tensor(batch)
        lo, hi = max(offset, start), min(offset + len(batch), end)
        if lo < hi:
            yield batch[lo - offset : hi - offset]
        offset += len(batch)
        if offset >= end:
            break
