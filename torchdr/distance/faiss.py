"""Distances based on Faiss backend."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
import warnings
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass, field

from torchdr.utils.faiss import faiss

LIST_METRICS_FAISS = ["euclidean", "sqeuclidean", "angular"]


@dataclass
class FaissConfig:
    """Configuration for FAISS k-NN computation.

    Parameters
    ----------
    use_float16 : bool, default=False
        Use float16 precision for GPU storage and computation. Reduces memory
        usage by ~50% and often improves performance on modern GPUs
        (compute capability 3.5+). Only applies to GPU mode.
    temp_memory : Union[str, float], default='auto'
        GPU temporary memory allocation in GB.
        - 'auto': Use FAISS default (~18% of GPU memory)
        - float/int: Explicit size in GB (e.g., 2.0 for 2GB)
        - 0: Disable pre-allocation (use cudaMalloc on demand)
        Only applies to GPU mode.
    device : int, default=0
        GPU device ID to use.
        Only applies when input is on CUDA.
    index_type : str, default='Flat'
        Type of FAISS index to use:
        - 'Flat': Exact brute-force search
        - 'IVF': Inverted file index for approximate search (requires nlist parameter)
        Currently only 'Flat' is fully supported.
    nprobe : int, default=1
        Number of clusters to search in IVF indexes. Higher values increase
        accuracy but decrease speed. Only used with index_type='IVF'.
    nlist : int, default=100
        Number of clusters for IVF indexes. Typical values range from
        sqrt(n) to 4*sqrt(n) where n is the dataset size.
        Only used with index_type='IVF'.

    Attributes
    ----------
    gpu_resources : Dict[int, Any]
        Dictionary mapping device IDs to their GPU resources (created on demand).

    Examples
    --------
    >>> # Basic float16 configuration for memory efficiency
    >>> config = FaissConfig(use_float16=True)

    >>> # GPU configuration with specific device
    >>> config = FaissConfig(device=1, use_float16=True)

    >>> # Custom memory allocation for large batch operations
    >>> config = FaissConfig(temp_memory=4.0, use_float16=True)  # 4GB temp memory

    >>> # Memory-constrained environment
    >>> config = FaissConfig(temp_memory=0.5, use_float16=True)  # 512MB only

    Notes
    -----
    - Float16 is recommended for datasets > 1M vectors or when GPU memory is limited
    - Increasing temp_memory helps with large batch operations but reduces memory
      available for data storage
    - IVF indexes trade accuracy for speed and are recommended for datasets > 10M vectors
    """

    use_float16: bool = False
    temp_memory: Union[str, float] = "auto"
    device: int = 0
    index_type: str = "Flat"
    nprobe: int = 1
    nlist: int = 100
    gpu_resources: Dict[int, Any] = field(default_factory=dict, init=False, repr=False)


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

    >>> # Use float16 for memory efficiency
    >>> config = FaissConfig(use_float16=True)
    >>> distances, indices = pairwise_distances_faiss(X, k=10, config=config)

    >>> # Multi-GPU configuration
    >>> config = FaissConfig(device=[0, 1], shard=True, use_float16=True)
    >>> distances, indices = pairwise_distances_faiss(X, k=10, config=config)

    >>> # Custom memory allocation for large batches
    >>> config = FaissConfig(temp_memory=4.0)  # 4GB temp memory
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
    n, d = X_np.shape

    if Y is None or Y is X:
        Y_np = X_np
        do_exclude = exclude_diag
    else:
        Y_np = Y.detach().cpu().numpy().astype(np.float32)
        do_exclude = False

    if config.index_type == "IVF":
        raise NotImplementedError(
            "[TorchDR] IVF indexes are not yet fully supported. "
            "Please use index_type='Flat' for now."
        )

    if metric == "angular":
        index = faiss.IndexFlatIP(d)
    elif metric in {"euclidean", "sqeuclidean"}:
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"[TorchDR] ERROR : Metric '{metric}' is not supported.")

    # Determine computation device
    if device == "auto":
        compute_device = X.device
    else:
        compute_device = torch.device(device)

    # Use GPU FAISS if requested and available
    if compute_device.type == "cuda":
        if hasattr(faiss, "StandardGpuResources"):
            index = _setup_gpu_index(index, config, d)
        else:
            warnings.warn(
                "[TorchDR] WARNING: `faiss-gpu` not installed, using CPU for Faiss computations. "
                "This may be slow. For faster performance, install `faiss-gpu`."
            )

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
    # With torchrun, each process handles exactly one GPU
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
        flat_config.useFloat16 = config.use_float16
        flat_config.device = device_id

        if isinstance(index, faiss.IndexFlatL2):
            gpu_index = faiss.GpuIndexFlatL2(res, d, flat_config)
        else:
            gpu_index = faiss.GpuIndexFlatIP(res, d, flat_config)
    else:
        gpu_index = faiss.index_cpu_to_gpu(res, device_id, index)

    return gpu_index
