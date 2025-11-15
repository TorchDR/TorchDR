"""Distributed training utilities for TorchDR.

This module provides automatic setup and cleanup for distributed training
when scripts are launched with torchrun or the TorchDR CLI, as well as
utilities for managing distributed multi-GPU computation.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import atexit
import os
from typing import Tuple

import torch
import torch.distributed as dist

__all__ = ["is_distributed", "get_rank", "get_world_size", "DistributedContext"]

_distributed_initialized_by_torchdr = False


def _auto_setup_distributed():
    """Automatically setup distributed training if launched with torchrun.

    This function is called on import and detects if the script was launched
    with torchrun (or the TorchDR CLI) by checking for the LOCAL_RANK environment
    variable. If found, it initializes the distributed process group for
    single-node multi-GPU training.
    """
    global _distributed_initialized_by_torchdr

    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # Initialize process group (single-node multi-GPU)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

        _distributed_initialized_by_torchdr = True

        # Register cleanup function
        atexit.register(_auto_cleanup_distributed)


def _auto_cleanup_distributed():
    """Automatically cleanup distributed training on exit.

    This function is registered with atexit and is called when the Python
    interpreter exits. It only destroys the process group if it was
    initialized by TorchDR's auto-setup.
    """
    global _distributed_initialized_by_torchdr

    if _distributed_initialized_by_torchdr and dist.is_initialized():
        dist.destroy_process_group()
        _distributed_initialized_by_torchdr = False


def is_distributed():
    """Check if distributed training is active.

    Returns
    -------
    bool
        True if running in distributed mode, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get the rank of the current process.

    Returns
    -------
    int
        Rank of the current process (0 for non-distributed).
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get the total number of processes.

    Returns
    -------
    int
        Total number of processes (1 for non-distributed).
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


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
    >>> from torchdr.distributed import DistributedContext
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

    @staticmethod
    def get_rank_for_indices(
        indices: torch.Tensor, n_samples: int, world_size: int
    ) -> torch.Tensor:
        """Compute which rank owns each global index.

        This is the inverse of compute_chunk_bounds: given global indices,
        returns which rank owns each index when n_samples is distributed
        across world_size ranks.

        Parameters
        ----------
        indices : torch.Tensor
            Global indices to look up (any shape).
        n_samples : int
            Total number of samples distributed across ranks.
        world_size : int
            Number of ranks/GPUs.

        Returns
        -------
        ranks : torch.Tensor
            Rank ownership for each index (same shape as indices).

        Examples
        --------
        >>> indices = torch.tensor([0, 25, 50, 75, 99])
        >>> ranks = DistributedContext.get_rank_for_indices(indices, 100, 4)
        >>> # Returns: tensor([0, 1, 2, 3, 3])
        """
        chunk_size = n_samples // world_size
        remainder = n_samples % world_size

        # Threshold where chunking strategy changes
        # First 'remainder' ranks get (chunk_size + 1) samples
        threshold = remainder * (chunk_size + 1)

        # For indices < threshold, each rank owns (chunk_size + 1) samples
        # For indices >= threshold, each rank owns chunk_size samples
        ranks = torch.where(
            indices < threshold,
            indices // (chunk_size + 1),  # Early ranks
            remainder + (indices - threshold) // chunk_size,  # Later ranks
        )

        # Clamp to valid rank range
        return torch.clamp(ranks, 0, world_size - 1)

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


# Auto-setup on import
_auto_setup_distributed()
