"""
Benchmark gradient compression methods for LargeVisMultiGPU.
Tests performance with no compression, fp16, and bf16.

Usage:
    torchrun --nproc_per_node=2 benchmark_gradient_compression.py
"""

import os
import time
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from torchdr import PCA
from torchdr.neighbor_embedding.largevis_multi_gpu import LargeVisMultiGPU


def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def benchmark_compression(x_tensor, compression_type, max_iter=1000):
    """Run LargeVis with specified compression and measure time.

    Parameters
    ----------
    x_tensor : torch.Tensor
        Input data on GPU
    compression_type : str or None
        Type of gradient compression: None, "fp16", "bf16"
    max_iter : int
        Number of iterations to run

    Returns
    -------
    embedding : torch.Tensor
        Final embedding
    elapsed_time : float
        Time taken in seconds
    """
    rank = dist.get_rank()

    if rank == 0:
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
        verbose=(rank == 0),
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
        print("Loading MNIST dataset...")

    # Load the MNIST dataset
    mnist = fetch_openml("mnist_784", cache=True, as_frame=False)
    x = mnist.data.astype("float32")

    # Perform PCA as preprocessing
    if rank == 0:
        print("Performing PCA preprocessing...")
    x = PCA(50).fit_transform(x)

    # Convert to torch tensor on GPU
    x_tensor = torch.from_numpy(x).cuda()

    if rank == 0:
        print(f"Data shape: {x_tensor.shape}")
        print(f"Running benchmarks with {world_size} GPUs...\n")

    # Test configurations
    compression_methods = [
        None,  # No compression (baseline)
        "fp16",  # Float16 compression
        "bf16",  # BFloat16 compression
    ]

    results = {}
    embeddings = {}

    # Run benchmarks
    for compression in compression_methods:
        z, elapsed = benchmark_compression(x_tensor, compression, max_iter=1000)

        # Store results
        compression_name = compression if compression else "none"
        results[compression_name] = elapsed
        embeddings[compression_name] = z.detach()

        if rank == 0:
            print(f"\nTime taken with {compression_name}: {elapsed:.2f} seconds")

        # Add delay between runs to let GPU cool down
        time.sleep(5)

    # Print summary on rank 0
    if rank == 0:
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"GPUs: {world_size}")
        print(f"Dataset size: {x_tensor.shape}")
        print("Iterations: 1000")
        print("-" * 60)

        # Calculate speedups
        baseline_time = results["none"]

        print(
            f"{'Method':<15} {'Time (s)':<12} {'Speedup':<12} {'Bandwidth Reduction':<20}"
        )
        print("-" * 60)

        for method in ["none", "fp16", "bf16"]:
            time_taken = results[method]
            speedup = baseline_time / time_taken
            bandwidth = "50%" if method in ["fp16", "bf16"] else "0%"

            method_display = method.upper() if method != "none" else "None (FP32)"
            print(
                f"{method_display:<15} {time_taken:<12.2f} {speedup:<12.2f}x {bandwidth:<20}"
            )

        print("-" * 60)

        # Compute embedding quality differences
        print("\nEmbedding Quality Check (L2 distance from baseline):")
        baseline_embedding = embeddings["none"]
        for method in ["fp16", "bf16"]:
            diff = torch.norm(embeddings[method] - baseline_embedding).item()
            print(f"{method.upper()}: {diff:.6f}")

        # Generate scatter plots for each compression method
        print("\nGenerating scatter plots...")

        # Convert to numpy for plotting
        embeddings_np = {
            method: embeddings[method].cpu().numpy() for method in embeddings
        }

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"LargeVis Embeddings - Gradient Compression Comparison ({world_size} GPUs)",
            fontsize=14,
        )

        # Get labels for coloring
        y_np = y.astype(int)

        # Plot each method
        for idx, (method, ax) in enumerate(zip(["none", "fp16", "bf16"], axes)):
            method_display = method.upper() if method != "none" else "FP32 (baseline)"
            embedding = embeddings_np[method]

            # Create scatter plot
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=y_np, cmap="tab10", s=1, alpha=0.5
            )

            # Add title and labels
            elapsed_time = results[method]
            ax.set_title(f"{method_display}\nTime: {elapsed_time:.1f}s", fontsize=12)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_aspect("equal")

            # Add grid
            ax.grid(True, alpha=0.3)

        # Add colorbar
        plt.colorbar(scatter, ax=axes.ravel().tolist(), label="Digit", shrink=0.6)

        plt.tight_layout()

        # Save the figure
        output_filename = f"gradient_compression_comparison_{world_size}gpus.png"
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"Scatter plots saved to {output_filename}")

        # Save results to file
        import json

        results_dict = {
            "world_size": world_size,
            "data_shape": list(x_tensor.shape),
            "iterations": 1000,
            "times": results,
            "speedups": {method: baseline_time / results[method] for method in results},
        }

        with open(f"gradient_compression_results_{world_size}gpus.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to gradient_compression_results_{world_size}gpus.json")

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
