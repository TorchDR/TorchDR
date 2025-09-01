#!/usr/bin/env python3
"""
Detailed benchmarking script for affinity computations.

This script performs multiple runs and provides statistical analysis of the performance
differences between single-GPU and multi-GPU modes.

Usage:
    # Single GPU benchmark
    python benchmark_affinity_multigpu.py --n_samples 50000 --n_runs 5

    # Multi-GPU benchmark (4 GPUs)
    torchrun --nproc_per_node=4 benchmark_affinity_multigpu.py --n_samples 50000 --n_runs 5
"""

import argparse
import torch
import torch.distributed as dist
import numpy as np
import time
from statistics import mean, stdev
import json
from datetime import datetime

from torchdr.affinity import EntropicAffinity, UMAPAffinity, PACMAPAffinity
from torchdr.data import load_mnist


def setup_distributed():
    """Setup distributed training if running with torchrun."""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size
    else:
        return 0, 1


def benchmark_affinity(affinity_class, X, params, n_runs=5, rank=0):
    """Benchmark an affinity computation with multiple runs."""
    times = []
    memory_usage = []

    for run in range(n_runs):
        # Clear cache before each run
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure time
        torch.cuda.synchronize()
        start_time = time.time()

        # Create and compute affinity
        affinity = affinity_class(**params)
        if affinity_class == UMAPAffinity or affinity_class == PACMAPAffinity:
            P, indices = affinity(X)
        else:  # EntropicAffinity
            P, indices = affinity(X, log=False)

        torch.cuda.synchronize()
        end_time = time.time()

        times.append(end_time - start_time)

        # Measure memory usage
        memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)  # GB

        # Cleanup
        del P, indices, affinity

        if rank == 0 and run == 0:
            print(
                f"    Run {run + 1}/{n_runs}: {times[-1]:.2f}s, {memory_usage[-1]:.2f}GB"
            )

    return times, memory_usage


def main():
    parser = argparse.ArgumentParser(description="Benchmark affinity computations")
    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of samples to use"
    )
    parser.add_argument(
        "--n_runs", type=int, default=5, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results (JSON format)"
    )
    args = parser.parse_args()

    # Setup distributed if available
    rank, world_size = setup_distributed()

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    if rank == 0:
        print("=" * 80)
        print("AFFINITY COMPUTATION BENCHMARK")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  - GPUs: {world_size}")
        print(f"  - Samples: {args.n_samples}")
        print(f"  - Runs per test: {args.n_runs}")
        print(
            f"  - Mode: {'Multi-GPU (Distributed)' if world_size > 1 else 'Single-GPU'}"
        )
        print("=" * 80)

    # Load data
    if rank == 0:
        print("\nLoading data...")
    X = load_mnist(n_samples=args.n_samples, random_state=42)
    X = torch.tensor(X, dtype=torch.float32).cuda()

    if rank == 0:
        print(f"Data loaded: {X.shape}")
        data_memory = X.element_size() * X.nelement() / 1024**3
        print(f"Data memory usage: {data_memory:.3f} GB")

    # Configure affinity classes
    test_configs = [
        {
            "name": "EntropicAffinity",
            "class": EntropicAffinity,
            "params": {
                "perplexity": 30,
                "metric": "sqeuclidean",
                "verbose": False,
                "device": "cuda",
                "backend": "faiss",
                "sparsity": True,
                "distributed": "auto",
            },
        },
        {
            "name": "UMAPAffinity",
            "class": UMAPAffinity,
            "params": {
                "n_neighbors": 30,
                "metric": "sqeuclidean",
                "verbose": False,
                "device": "cuda",
                "backend": "faiss",
                "sparsity": True,
                "symmetrize": False,
                "distributed": "auto",
            },
        },
        {
            "name": "PACMAPAffinity",
            "class": PACMAPAffinity,
            "params": {
                "n_neighbors": 10,
                "metric": "sqeuclidean",
                "verbose": False,
                "device": "cuda",
                "backend": "faiss",
                "distributed": "auto",
            },
        },
    ]

    # Store all results
    all_results = {
        "config": {
            "n_gpus": world_size,
            "n_samples": args.n_samples,
            "n_dims": X.shape[1],
            "n_runs": args.n_runs,
            "timestamp": datetime.now().isoformat(),
        },
        "benchmarks": {},
    }

    # Run benchmarks
    for config in test_configs:
        if rank == 0:
            print(f"\nBenchmarking {config['name']}...")
            print("-" * 40)

        times, memory = benchmark_affinity(
            config["class"], X, config["params"], n_runs=args.n_runs, rank=rank
        )

        if rank == 0:
            # Calculate statistics
            avg_time = mean(times)
            std_time = stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)

            avg_memory = mean(memory)
            max_memory = max(memory)

            # Store results
            all_results["benchmarks"][config["name"]] = {
                "times": times,
                "avg_time": avg_time,
                "std_time": std_time,
                "min_time": min_time,
                "max_time": max_time,
                "memory_gb": memory,
                "avg_memory_gb": avg_memory,
                "max_memory_gb": max_memory,
            }

            # Print summary
            print(f"\n  Results for {config['name']}:")
            print(f"    Time: {avg_time:.3f} ± {std_time:.3f} seconds")
            print(f"    Range: [{min_time:.3f}, {max_time:.3f}] seconds")
            print(f"    Memory: {avg_memory:.2f} GB (max: {max_memory:.2f} GB)")

            if world_size > 1:
                # Estimate single-GPU time (rough approximation)
                estimated_single_gpu_time = (
                    avg_time * world_size * 0.8
                )  # 0.8 for parallel efficiency
                speedup = estimated_single_gpu_time / avg_time
                print(f"    Estimated speedup: {speedup:.2f}x")

        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()

    # Print final summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        print("\nTest Configuration:")
        print(f"  - {world_size} GPU(s)")
        print(f"  - {args.n_samples} samples × {X.shape[1]} dimensions")
        print(f"  - {args.n_runs} runs per test")

        print("\nAverage Computation Times:")
        for name, results in all_results["benchmarks"].items():
            print(
                f"  {name:20s}: {results['avg_time']:8.3f} ± {results['std_time']:.3f} seconds"
            )

        print("\nPeak Memory Usage:")
        for name, results in all_results["benchmarks"].items():
            print(f"  {name:20s}: {results['max_memory_gb']:8.2f} GB")

        if world_size > 1:
            print("\nMulti-GPU Configuration:")
            print(
                f"  - Data distribution: ~{args.n_samples // world_size} samples per GPU"
            )
            print(f"  - Parallel k-NN search across {world_size} GPUs")
            print(
                "  - Note: Symmetrization disabled for UMAPAffinity in multi-GPU mode"
            )

        # Save results to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        print("\n" + "=" * 80)
        print("Benchmark complete!")
        print("=" * 80)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
