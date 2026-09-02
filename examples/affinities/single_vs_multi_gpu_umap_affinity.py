#!/usr/bin/env python3
"""
Single vs Multi-GPU UMAPAffinity Comparison
============================================

Simple example of using UMAPAffinity with multi-GPU on 10x mouse Zheng dataset.
Compares distributed vs non-distributed performance and verifies that the rows
owned by rank 0 match the single-GPU result.

TorchDR initializes the process group itself when the script is launched with
``torchrun`` or the ``torchdr`` CLI, so no distributed boilerplate is required
here. Calling ``torch.distributed.init_process_group`` after importing TorchDR
would raise ``ValueError: trying to initialize the default process group
twice!``; use :func:`torchdr.distributed.init_distributed` if a script needs to
create the group explicitly.

Usage:
    torchdr --gpus 4 single_vs_multi_gpu_umap_affinity.py
    torchrun --nproc_per_node=4 single_vs_multi_gpu_umap_affinity.py
"""

import torch
import torch.distributed as dist
import time
import gzip
import pickle
from io import BytesIO
import requests

from torchdr.affinity import UMAPAffinity
from torchdr.distributed import get_rank, get_world_size, is_distributed


def download_and_load_dataset(url):
    """Download and load pickled dataset from URL."""
    response = requests.get(url)
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
        data = pickle.load(f)
    return data


def sparse_entries(values, indices, row_offset, n_total):
    """Return sorted (encoded coordinate, value) pairs of a padded sparse chunk.

    Rows are packed with ``-1`` index padding, and the number of stored columns
    per row depends on the local edge distribution. Encoding the coordinates
    makes chunks with different padding widths directly comparable.
    """
    rows = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
    keys = (rows + row_offset) * n_total + indices
    mask = indices >= 0
    keys, values = keys[mask], values[mask]
    order = torch.argsort(keys)
    return keys[order], values[order]


def main():
    # TorchDR sets up the process group on import under torchrun / the CLI.
    rank = get_rank()
    world_size = get_world_size()

    # Print from all ranks to verify all GPUs are active
    print(
        f"[Rank {rank}] Process started on GPU {torch.cuda.current_device()}, device name: {torch.cuda.get_device_name()}"
    )
    if is_distributed():
        dist.barrier()  # Synchronize before continuing

    if rank == 0:
        print(f"\nRunning comparison on {world_size} GPUs")
        print("Loading 10x mouse Zheng dataset...")

    # Download and load 10x mouse Zheng data
    url_10x = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
    data_10x = download_and_load_dataset(url_10x)

    # Data is already PCA-reduced to 50 dimensions
    x = data_10x["pca_50"].astype("float32")

    if rank == 0:
        print("Data already PCA-reduced to 50 dimensions")

    # Convert to tensor (data stays on CPU)
    X = torch.tensor(x, dtype=torch.float32)

    if rank == 0:
        print(f"Data shape: {X.shape}")
        print("\n" + "=" * 60)
        print("TEST 1: Multi-GPU UMAPAffinity (distributed=True)")
        print("=" * 60)
        start_time = time.time()

    # Create UMAPAffinity with distributed=True
    affinity_distributed = UMAPAffinity(
        n_neighbors=30,
        metric="sqeuclidean",
        verbose=(rank == 0),
        device="cuda",
        backend="faiss",
        sparsity=True,
        distributed=True,  # Force distributed mode
    )

    # Compute affinity matrix with distributed mode
    P_dist, indices_distributed = affinity_distributed(X, log=False)
    chunk_start = affinity_distributed.chunk_start_

    # Synchronize all GPUs
    if is_distributed():
        dist.barrier()

    # Only rank 0 computes the single-GPU reference and reports the comparison.
    if rank != 0:
        return

    distributed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("TEST 2: Single-GPU UMAPAffinity (distributed=False)")
    print("=" * 60)
    start_time = time.time()

    # Same settings as the distributed run, so both compute the symmetrized
    # affinity P + Pᵀ - P∘Pᵀ and the outputs are directly comparable.
    affinity_single = UMAPAffinity(
        n_neighbors=30,
        metric="sqeuclidean",
        verbose=True,
        device="cuda",
        backend="faiss",
        sparsity=True,
        distributed=False,  # Force single-GPU mode
    )

    # Compute affinity matrix with single GPU
    P_single, indices_single = affinity_single(X, log=False)

    single_time = time.time() - start_time
    print(f"\nSingle-GPU computation completed in {single_time:.2f} seconds")

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    # Timing comparison
    print("\nTiming:")
    print(f"  Multi-GPU ({world_size} GPUs): {distributed_time:.2f} seconds")
    print(f"  Single-GPU: {single_time:.2f} seconds")
    print(f"  Speedup: {single_time / distributed_time:.2f}x")

    # Output shape comparison
    print("\nOutput shapes:")
    print(f"  Multi-GPU affinity shape: {P_dist.shape}")
    print(f"  Single-GPU affinity shape: {P_single.shape}")
    print(f"  Multi-GPU indices shape: {indices_distributed.shape}")
    print(f"  Single-GPU indices shape: {indices_single.shape}")

    print("\nAffinity statistics (Multi-GPU):")
    print(f"  Min value: {P_dist.min().item():.6e}")
    print(f"  Max value: {P_dist.max().item():.6e}")
    print(f"  Mean value: {P_dist.mean().item():.6e}")

    print("\nAffinity statistics (Single-GPU):")
    print(f"  Min value: {P_single.min().item():.6e}")
    print(f"  Max value: {P_single.max().item():.6e}")
    print(f"  Mean value: {P_single.mean().item():.6e}")

    # Symmetrization gives each row a different number of stored neighbors, so
    # the two runs are compared as coordinate/value pairs over rank 0's rows.
    chunk_size = P_dist.shape[0]
    n_total = P_single.shape[0]
    keys_dist, values_dist = sparse_entries(
        P_dist, indices_distributed, chunk_start, n_total
    )
    keys_single, values_single = sparse_entries(
        P_single[:chunk_size], indices_single[:chunk_size], 0, n_total
    )

    print(f"\nOutput similarity check (comparing rank 0's chunk of {chunk_size} rows):")
    print(
        f"  Stored entries (multi-GPU / single-GPU): {keys_dist.numel()}"
        f" / {keys_single.numel()}"
    )

    same_support = keys_dist.numel() == keys_single.numel() and bool(
        (keys_dist == keys_single).all()
    )
    print(f"  Same sparsity pattern: {same_support}")

    if same_support:
        abs_diff = (values_dist - values_single).abs()
        print(f"  Mean absolute difference: {abs_diff.mean().item():.6e}")
        print(f"  Max absolute difference: {abs_diff.max().item():.6e}")
        print(
            "  Affinity values match (rtol=1e-3): "
            f"{torch.allclose(values_dist, values_single, rtol=1e-3, atol=1e-6)}"
        )


if __name__ == "__main__":
    main()
