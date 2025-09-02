#!/usr/bin/env python3
"""
Simple example of using UMAPAffinity with multi-GPU on 10x mouse Zheng dataset.
Compares distributed vs non-distributed performance and verifies outputs are similar.
Must be launched with torchrun for distributed execution.

Usage:
    torchrun --nproc_per_node=4 test_mnist_entropic_multigpu.py
"""

import os
import torch
import torch.distributed as dist
import time
import gzip
import pickle
from io import BytesIO
import requests

from torchdr.affinity import UMAPAffinity


def download_and_load_dataset(url):
    """Download and load pickled dataset from URL."""
    response = requests.get(url)
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
        data = pickle.load(f)
    return data


def setup_distributed():
    """Initialize distributed training environment."""
    # torchrun sets the environment variables, but we still need to init the process group
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Initialize distributed training
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Print from all ranks to verify all GPUs are active
    print(
        f"[Rank {rank}] Process started on GPU {torch.cuda.current_device()}, device name: {torch.cuda.get_device_name()}"
    )
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
        print(f"Data already PCA-reduced to 50 dimensions")

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

    # Synchronize all GPUs
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        distributed_time = time.time() - start_time
        print(f"\nMulti-GPU computation completed in {distributed_time:.2f} seconds")

        print("\n" + "=" * 60)
        print("TEST 2: Single-GPU UMAPAffinity (distributed=False)")
        print("=" * 60)
        start_time = time.time()

    # Only rank 0 computes the single-GPU version
    if rank == 0:
        # Create UMAPAffinity with distributed=False
        affinity_single = UMAPAffinity(
            n_neighbors=30,
            metric="sqeuclidean",
            verbose=True,
            device="cuda",
            backend="faiss",
            sparsity=True,
            symmetrize=False,  # Match multi-GPU behavior
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
        print(f"\nTiming:")
        print(f"  Multi-GPU ({world_size} GPUs): {distributed_time:.2f} seconds")
        print(f"  Single-GPU: {single_time:.2f} seconds")
        print(f"  Speedup: {single_time / distributed_time:.2f}x")

        # Output shape comparison
        print(f"\nOutput shapes:")
        print(f"  Multi-GPU affinity shape: {P_dist.shape}")
        print(f"  Single-GPU affinity shape: {P_single.shape}")
        print(f"  Multi-GPU indices shape: {indices_distributed.shape}")
        print(f"  Single-GPU indices shape: {indices_single.shape}")

        print(f"\nAffinity statistics (Multi-GPU):")
        print(f"  Min value: {P_dist.min().item():.6e}")
        print(f"  Max value: {P_dist.max().item():.6e}")
        print(f"  Mean value: {P_dist.mean().item():.6e}")

        print(f"\nAffinity statistics (Single-GPU):")
        print(f"  Min value: {P_single.min().item():.6e}")
        print(f"  Max value: {P_single.max().item():.6e}")
        print(f"  Mean value: {P_single.mean().item():.6e}")

        # Check similarity of outputs (P_dist is chunked, P_single is full)
        print(f"\nOutput similarity check (comparing rank 0's chunk):")

        # Get the size of rank 0's chunk
        chunk_size = P_dist.shape[0]

        # Compare indices for the chunk
        indices_match = torch.allclose(
            indices_distributed, indices_single[:chunk_size], rtol=1e-5
        )
        print(f"  Indices match (chunk of {chunk_size} points): {indices_match}")

        # Compare affinity values for the chunk
        P_single_chunk = P_single[:chunk_size]
        values_diff = torch.abs(P_dist - P_single_chunk).mean()
        print(
            f"  Mean absolute difference in affinity values: {values_diff.item():.6e}"
        )

        # Check relative difference
        relative_diff = (
            torch.abs(P_dist - P_single_chunk) / (P_single_chunk + 1e-10)
        ).mean()
        print(f"  Mean relative difference: {relative_diff.item():.6e}")

        values_match = torch.allclose(P_dist, P_single_chunk, rtol=1e-3, atol=1e-6)
        print(f"  Affinity values match (rtol=1e-3): {values_match}")

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
