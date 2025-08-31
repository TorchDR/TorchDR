"""
Benchmark gradient compression methods for LargeVisMultiGPU on 10x Mouse Zheng dataset.
Tests performance with no compression, fp16, and bf16.

Usage:
    torchrun --nproc_per_node=2 benchmark_gradient_compression_10x.py
"""

import os
import time
import gzip
import pickle
from io import BytesIO
import requests
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from torchdr.neighbor_embedding.largevis_multi_gpu import LargeVisMultiGPU


def setup_distributed():
    """Initialize distributed training environment."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def download_and_load_dataset(url):
    """Download and load the 10x dataset from URL."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(BytesIO(response.content), "rb") as f:
        data = pickle.load(f)
    return data


def benchmark_compression(
    x_tensor, compression_type, max_iter=1000, verbose_rank_0=True
):
    """Run LargeVis with specified compression and measure time.

    Parameters
    ----------
    x_tensor : torch.Tensor
        Input data on GPU
    compression_type : str or None
        Type of gradient compression: None, "fp16", "bf16"
    max_iter : int
        Number of iterations to run
    verbose_rank_0 : bool
        Whether rank 0 should print progress

    Returns
    -------
    embedding : torch.Tensor
        Final embedding
    elapsed_time : float
        Time taken in seconds
    """
    rank = dist.get_rank()

    if rank == 0 and verbose_rank_0:
        print(f"\n{'=' * 60}")
        print(
            f"Testing gradient compression: {compression_type if compression_type else 'None (fp32)'}"
        )
        print(f"{'=' * 60}")

    # Create model with specified compression
    largevis = LargeVisMultiGPU(
        perplexity=30,
        n_components=2,
        lr="auto",
        max_iter=max_iter,
        verbose=(rank == 0 and verbose_rank_0),
        random_state=42,
        n_negatives=5,
        sparsity=True,
        compile=False,  # Keep consistent across tests
        gradient_compression=compression_type,
    )

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

    start_time = time.perf_counter()

    # Run fit_transform
    z = largevis.fit_transform(x_tensor)

    # Synchronize after computation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

    elapsed_time = time.perf_counter() - start_time

    return z, elapsed_time


def main():
    # Initialize distributed training
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Running gradient compression benchmark on {world_size} GPUs")
        print("Loading 10x Mouse Zheng single-cell dataset...")

    # Download and load the dataset
    url = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"

    try:
        data = download_and_load_dataset(url)
        # Use PCA-reduced data like in the original benchmark
        X = data["pca_50"]
        y = data["CellType1"]

        # Convert labels to numeric
        unique_labels = np.unique(y)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_to_idx[label] for label in y])
    except Exception as e:
        if rank == 0:
            print(f"\nError loading dataset: {e}")
            print(f"Failed to download from: {url}")
            print("\nPlease check your internet connection or the dataset URL.")
        cleanup_distributed()
        raise RuntimeError(f"Dataset download failed: {e}")

    # Convert to torch tensor on GPU
    x_tensor = torch.from_numpy(X.astype(np.float32)).cuda()

    if rank == 0:
        print(f"Data shape: {x_tensor.shape}")
        print(f"Data type: {x_tensor.dtype}")
        print(f"Number of unique labels: {len(np.unique(y))}")
        print(f"Running benchmarks with {world_size} GPUs...\n")

    # Test configurations
    compression_methods = [
        None,  # No compression (baseline)
        "fp16",  # Float16 compression
        "bf16",  # BFloat16 compression
    ]

    results = {}
    embeddings = {}

    # Warmup run (not timed)
    if rank == 0:
        print("Running warmup...")
    _ = benchmark_compression(x_tensor, None, max_iter=50, verbose_rank_0=False)
    time.sleep(2)

    # Run benchmarks
    for compression in compression_methods:
        z, elapsed = benchmark_compression(
            x_tensor, compression, max_iter=1000, verbose_rank_0=True
        )

        # Store results
        compression_name = compression if compression else "none"
        results[compression_name] = elapsed
        embeddings[compression_name] = z.detach()

        if rank == 0:
            print(f"\nTime taken with {compression_name}: {elapsed:.2f} seconds")
            print(f"Throughput: {1000 / elapsed:.2f} iterations/second")

        # Add delay between runs to let GPU cool down
        time.sleep(3)

    # Print summary on rank 0
    if rank == 0:
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY - 10X DATASET")
        print("=" * 60)
        print(f"GPUs: {world_size}")
        print(f"Dataset size: {x_tensor.shape}")
        print("Iterations: 1000")
        print("-" * 60)

        # Calculate speedups
        baseline_time = results["none"]

        print(f"{'Method':<15} {'Time (s)':<12} {'Speedup':<12} {'Throughput':<20}")
        print("-" * 60)

        for method in ["none", "fp16", "bf16"]:
            time_taken = results[method]
            speedup = baseline_time / time_taken
            throughput = 1000 / time_taken

            method_display = method.upper() if method != "none" else "None (FP32)"
            print(
                f"{method_display:<15} {time_taken:<12.2f} {speedup:<12.2f}x {throughput:<20.2f} iter/s"
            )

        print("-" * 60)

        # Compute embedding quality differences
        print("\nEmbedding Quality Check (L2 distance from baseline):")
        baseline_embedding = embeddings["none"]
        for method in ["fp16", "bf16"]:
            diff = torch.norm(embeddings[method] - baseline_embedding).item()
            relative_diff = diff / torch.norm(baseline_embedding).item()
            print(f"{method.upper()}: {diff:.6f} (relative: {relative_diff:.6%})")

        # Generate scatter plots for each compression method
        print("\nGenerating scatter plots...")

        # Convert to numpy for plotting
        embeddings_np = {
            method: embeddings[method].cpu().numpy() for method in embeddings
        }

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"LargeVis on 10X Dataset - Gradient Compression Comparison ({world_size} GPUs)",
            fontsize=14,
        )

        # Use labels for coloring (convert to int for colormap)
        y_int = y.astype(int) if y.dtype != np.int64 else y
        unique_labels = np.unique(y_int)
        n_colors = len(unique_labels)

        # Choose appropriate colormap
        if n_colors <= 20:
            cmap = "tab20"
        else:
            cmap = "viridis"

        # Plot each method
        for idx, (method, ax) in enumerate(zip(["none", "fp16", "bf16"], axes)):
            method_display = method.upper() if method != "none" else "FP32 (baseline)"
            embedding = embeddings_np[method]

            # Create scatter plot
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=y_int,
                cmap=cmap,
                s=0.5,  # Smaller points for larger dataset
                alpha=0.6,
                rasterized=True,  # For faster rendering
            )

            # Add title and labels
            elapsed_time = results[method]
            speedup = baseline_time / elapsed_time if method != "none" else 1.0
            ax.set_title(
                f"{method_display}\nTime: {elapsed_time:.1f}s | Speedup: {speedup:.2f}x",
                fontsize=11,
            )
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_aspect("equal")

            # Add grid
            ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(
            scatter, ax=axes.ravel().tolist(), label="Cell Type", shrink=0.6
        )
        cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()

        # Save the figure
        output_filename = f"gradient_compression_10x_comparison_{world_size}gpus.png"
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"Scatter plots saved to {output_filename}")

        # Additional statistics
        print("\nCommunication cost analysis:")
        embedding_bytes = (
            x_tensor.shape[0] * 2 * 4
        )  # n_points * n_components * sizeof(float32)
        print(f"Gradient size per all_reduce: {embedding_bytes / 1024**2:.2f} MB")
        print(f"FP16/BF16 reduces to: {embedding_bytes / 1024**2 / 2:.2f} MB")

        total_comm_baseline = embedding_bytes * 1000 / 1024**3  # Total GB communicated
        print(f"Total communication (FP32): {total_comm_baseline:.2f} GB")
        print(f"Total communication (FP16/BF16): {total_comm_baseline / 2:.2f} GB")

        # Save results to file
        import json

        results_dict = {
            "world_size": world_size,
            "data_shape": list(x_tensor.shape),
            "iterations": 1000,
            "times": results,
            "speedups": {method: baseline_time / results[method] for method in results},
            "throughput": {method: 1000 / results[method] for method in results},
            "embedding_quality": {
                method: (
                    torch.norm(embeddings[method] - embeddings["none"]).item()
                    if method != "none"
                    else 0.0
                )
                for method in ["none", "fp16", "bf16"]
            },
        }

        with open(f"gradient_compression_10x_results_{world_size}gpus.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        print(
            f"\nResults saved to gradient_compression_10x_results_{world_size}gpus.json"
        )

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
