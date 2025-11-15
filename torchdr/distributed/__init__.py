"""Distributed training utilities for TorchDR.

This module provides automatic setup and cleanup for distributed training
when scripts are launched with torchrun or the TorchDR CLI.
"""

import atexit
import os

import torch
import torch.distributed as dist

__all__ = ["is_distributed", "get_rank", "get_world_size"]

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


# Auto-setup on import
_auto_setup_distributed()
