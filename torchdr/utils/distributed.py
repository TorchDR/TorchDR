"""Distributed computation utilities for multi-GPU operations."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os
import torch
from typing import Tuple


class DistributedContext:
    """Context manager for distributed multi-GPU computation.

    This class handles the setup and management of distributed computation
    across multiple GPUs, particularly for k-NN distance computations.
    Each GPU computes its assigned chunk of rows from the full dataset.

    Parameters
    ----------
    force_enable : bool, default=False
        Force enable distributed mode even if torch.distributed is not
        initialized. Useful for testing.

    Attributes
    ----------
    is_initialized : bool
        Whether distributed mode is active.
    rank : int
        Global rank of this process (0 to world_size-1).
    world_size : int
        Total number of processes in the distributed group.
    local_rank : int
        Local rank on this node (used for CUDA device assignment).

    Examples
    --------
    >>> import torch
    >>> from torchdr.utils import DistributedContext
    >>> from torchdr.distance import pairwise_distances
    >>>
    >>> # Initialize distributed (usually done by launcher like torchrun)
    >>> # torch.distributed.init_process_group(...)
    >>>
    >>> # Create distributed context
    >>> dist_ctx = DistributedContext()
    >>>
    >>> if dist_ctx.is_initialized:
    ...     print(f"Rank {dist_ctx.rank}/{dist_ctx.world_size}")
    ...     # Each GPU computes its chunk
    ...     distances, indices = pairwise_distances(
    ...         X, k=10, distributed_ctx=dist_ctx
    ...     )
    ...     print(f"Computed {distances.shape[0]} rows on this GPU")

    Notes
    -----
    - Automatically sets CUDA device to local_rank when initialized
    - Each rank computes a roughly equal chunk of rows
    - Results remain distributed (no automatic gathering)
    - Requires sparsity (k-NN) for efficient distributed computation
    """

    def __init__(self, force_enable: bool = False):
        dist_initialized = torch.distributed.is_initialized()
        self.is_initialized = dist_initialized or force_enable

        if dist_initialized:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

    def compute_chunk_bounds(self, n_samples: int) -> Tuple[int, int]:
        """Compute start and end indices for this rank's chunk.

        Divides n_samples as evenly as possible across all ranks.
        If n_samples is not evenly divisible, the first (n_samples % world_size)
        ranks get one extra sample.

        Parameters
        ----------
        n_samples : int
            Total number of samples to divide.

        Returns
        -------
        chunk_start : int
            Starting index (inclusive) for this rank.
        chunk_end : int
            Ending index (exclusive) for this rank.

        Examples
        --------
        >>> dist_ctx = DistributedContext()
        >>> # With 100 samples and 4 GPUs:
        >>> # Rank 0: [0:25], Rank 1: [25:50], Rank 2: [50:75], Rank 3: [75:100]
        >>> start, end = dist_ctx.compute_chunk_bounds(100)
        """
        chunk_size = n_samples // self.world_size
        remainder = n_samples % self.world_size

        if self.rank < remainder:
            chunk_start = self.rank * (chunk_size + 1)
            chunk_end = chunk_start + chunk_size + 1
        else:
            chunk_start = self.rank * chunk_size + remainder
            chunk_end = chunk_start + chunk_size

        return chunk_start, chunk_end

    def get_faiss_config(self, base_config=None):
        """Create FaissConfig for this rank's GPU device.

        If a base_config is provided, copies all settings but overrides
        the device to use this rank's local GPU.

        Parameters
        ----------
        base_config : FaissConfig, optional
            Base configuration to copy settings from. If None, creates
            a default config for this rank's device.

        Returns
        -------
        FaissConfig
            Configuration object with device set to this rank's local GPU.

        Examples
        --------
        >>> dist_ctx = DistributedContext()
        >>> # Use defaults for this GPU
        >>> config = dist_ctx.get_faiss_config()
        >>>
        >>> # Copy user config but override device
        >>> user_config = FaissConfig(temp_memory=2.0, index_type="IVF")
        >>> config = dist_ctx.get_faiss_config(user_config)
        """
        # Import here to avoid circular dependency
        from torchdr.distance import FaissConfig

        if base_config is None:
            return FaissConfig(device=self.local_rank)
        else:
            return FaissConfig(
                temp_memory=base_config.temp_memory,
                device=self.local_rank,
                index_type=base_config.index_type,
                nprobe=base_config.nprobe,
                nlist=base_config.nlist,
                **base_config.faiss_kwargs,
            )

    def __repr__(self):
        if self.is_initialized:
            return (
                f"DistributedContext(rank={self.rank}, world_size={self.world_size}, "
                f"local_rank={self.local_rank})"
            )
        else:
            return "DistributedContext(not initialized)"
