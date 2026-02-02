"""Benchmark FAISS index types (Flat, IVF, IVFPQ) on single and multi-GPU.

Usage:
    python benchmark_faiss.py --sizes 1000000 10000000
    torchrun --nproc_per_node=4 benchmark_faiss.py --sizes 1000000 10000000
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from torchdr.distance import pairwise_distances, FaissConfig
from torchdr.utils import DistributedContext

CONFIGS = {
    "Flat": FaissConfig(index_type="Flat"),
    "IVF-np10": FaissConfig(index_type="IVF", nprobe=10),
    "IVF-np50": FaissConfig(index_type="IVF", nprobe=50),
    "IVFPQ-M16-np10": FaissConfig(index_type="IVFPQ", M=16, nprobe=10),
    "IVFPQ-M16-np50": FaissConfig(index_type="IVFPQ", M=16, nprobe=50),
}


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    return torch.cuda.max_memory_allocated() / 1024**2


def benchmark_config(X, dataloader, config, k, n_runs, distributed_ctx):
    """Benchmark a single config, return (time, memory_mb) or None on failure."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

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

    mean_time = sum(times) / len(times)
    peak_mem = get_gpu_memory_mb()
    return mean_time, peak_mem


def run_benchmark(n_samples, n_features, k, n_runs, distributed_ctx):
    """Run benchmark for all configs on a dataset."""
    rank = distributed_ctx.rank if distributed_ctx else 0
    n_gpus = distributed_ctx.world_size if distributed_ctx else 1

    # Generate data
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)

    if distributed_ctx:
        dataloader = DataLoader(TensorDataset(X), batch_size=50000, shuffle=False)
        X_gpu = None
    else:
        dataloader = None
        X_gpu = X.cuda()

    results = []
    for name, config in CONFIGS.items():
        if rank == 0:
            print(f"  {name:<20}", end="", flush=True)

        try:
            mean_time, peak_mem = benchmark_config(
                X_gpu, dataloader, config, k, n_runs, distributed_ctx
            )
            throughput = n_samples / mean_time / 1e6

            if rank == 0:
                print(f"{mean_time:>7.2f}s  {peak_mem:>8.0f}MB  {throughput:>5.2f}M/s")

            results.append(
                {
                    "config": name,
                    "samples": n_samples,
                    "gpus": n_gpus,
                    "time": mean_time,
                    "memory_mb": peak_mem,
                    "throughput": throughput,
                }
            )
        except Exception as e:
            if rank == 0:
                print(f"FAILED: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000_000, 10_000_000])
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    # Setup distributed if running with torchrun
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    ctx = None
    if distributed:
        ctx = DistributedContext()
        ctx.init()

    rank = ctx.rank if ctx else 0
    n_gpus = ctx.world_size if ctx else 1

    if rank == 0:
        print(
            f"FAISS Benchmark | GPU: {torch.cuda.get_device_name(0)} | GPUs: {n_gpus}"
        )
        print("=" * 60)

    all_results = []
    for n in args.sizes:
        if rank == 0:
            print(f"\n{n:,} samples x {args.features}D:")
            print(f"  {'Config':<20}{'Time':>8}  {'Memory':>9}  {'Speed':>7}")
            print("  " + "-" * 50)

        all_results.extend(run_benchmark(n, args.features, args.k, args.runs, ctx))

    # Summary
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(
            f"{'Config':<20} {'Samples':>10} {'GPUs':>5} {'Time':>8} {'Mem(MB)':>8} {'M/s':>6}"
        )
        print("-" * 60)
        for r in all_results:
            print(
                f"{r['config']:<20} {r['samples']:>10,} {r['gpus']:>5} "
                f"{r['time']:>8.2f} {r['memory_mb']:>8.0f} {r['throughput']:>6.2f}"
            )


if __name__ == "__main__":
    main()
