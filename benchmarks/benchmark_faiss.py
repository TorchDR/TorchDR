"""Benchmark FAISS index types (Flat, IVF, IVFPQ) on single and multi-GPU.

Usage:
    # Single GPU
    python benchmark_faiss.py --sizes 1000000 10000000

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 benchmark_faiss.py --sizes 1000000 10000000
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchdr.distance import pairwise_distances, FaissConfig
from torchdr.utils import DistributedContext


def benchmark(
    n_samples: int,
    n_features: int,
    k: int = 15,
    n_runs: int = 3,
    distributed_ctx=None,
):
    """Run benchmark for all FAISS configurations."""
    rank = distributed_ctx.rank if distributed_ctx else 0
    world_size = distributed_ctx.world_size if distributed_ctx else 1

    configs = {
        "Flat": FaissConfig(index_type="Flat"),
        "IVF-nprobe10": FaissConfig(index_type="IVF", nprobe=10),
        "IVF-nprobe50": FaissConfig(index_type="IVF", nprobe=50),
        "IVFPQ-M16-nprobe10": FaissConfig(index_type="IVFPQ", M=16, nprobe=10),
        "IVFPQ-M16-nprobe50": FaissConfig(index_type="IVFPQ", M=16, nprobe=50),
    }

    # Generate data
    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {n_samples:,} x {n_features} | GPUs: {world_size}")
        print(f"{'=' * 70}")

    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)

    if distributed_ctx:
        dataloader = DataLoader(
            TensorDataset(X), batch_size=50000, shuffle=False, num_workers=2
        )
    else:
        X = X.cuda()

    results = []
    for name, config in configs.items():
        if rank == 0:
            print(f"  {name:<25}", end="", flush=True)

        try:
            times = []
            for _ in range(n_runs):
                if distributed_ctx:
                    import torch.distributed as dist

                    dist.barrier()

                torch.cuda.synchronize()
                start = time.perf_counter()

                if distributed_ctx:
                    pairwise_distances(
                        dataloader,
                        k=k,
                        backend=config,
                        return_indices=True,
                        distributed_ctx=distributed_ctx,
                    )
                else:
                    pairwise_distances(X, k=k, backend=config, return_indices=True)

                torch.cuda.synchronize()
                if distributed_ctx:
                    dist.barrier()

                times.append(time.perf_counter() - start)
                torch.cuda.empty_cache()

            mean_t, std_t = np.mean(times), np.std(times)
            throughput = n_samples / mean_t / 1e6

            if rank == 0:
                print(f"{mean_t:>8.2f}s (±{std_t:.2f})  {throughput:>6.2f} Mvec/s")

            results.append(
                {
                    "config": name,
                    "n_samples": n_samples,
                    "n_gpus": world_size,
                    "mean_time": mean_t,
                    "throughput": throughput,
                }
            )

        except Exception as e:
            if rank == 0:
                print(f"  FAILED: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="FAISS benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000000, 10000000])
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    # Check distributed
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    ctx = None

    if is_distributed:
        ctx = DistributedContext()
        ctx.init()

    rank = ctx.rank if ctx else 0

    if rank == 0:
        print("FAISS Benchmark")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Distributed: {is_distributed}")

    all_results = []
    for n_samples in args.sizes:
        results = benchmark(
            n_samples, args.features, args.k, args.runs, distributed_ctx=ctx
        )
        all_results.extend(results)

    # Summary table
    if rank == 0:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(
            f"{'Config':<25} {'Samples':>12} {'GPUs':>6} {'Time (s)':>10} {'Mvec/s':>10}"
        )
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['config']:<25} {r['n_samples']:>12,} {r['n_gpus']:>6} "
                f"{r['mean_time']:>10.2f} {r['throughput']:>10.2f}"
            )


if __name__ == "__main__":
    main()
