"""Benchmark FAISS index types: Flat, IVF, and IVFPQ.

Measures query time and recall@k (accuracy vs exact Flat index).
Outputs results to CSV for plotting.

Usage:
    python run_benchmark.py --n_samples 1000000 --output results.csv
"""

import argparse
import csv
import time
import torch
import numpy as np

from torchdr.distance import pairwise_distances, FaissConfig


def compute_recall(indices_true: torch.Tensor, indices_approx: torch.Tensor) -> float:
    """Compute recall@k: fraction of true neighbors found."""
    k = indices_true.shape[1]
    matches = (indices_true.unsqueeze(2) == indices_approx.unsqueeze(1)).any(dim=2)
    return matches.float().sum(dim=1).mean().item() / k


def benchmark_config(X, config, k, ground_truth=None, query_indices=None):
    """Benchmark a config, return (time_seconds, recall, memory_mb).

    If query_indices is provided, computes recall only on those samples
    (for large datasets where full ground truth is infeasible).
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()
    _, indices = pairwise_distances(
        X, k=k, backend=config, return_indices=True, exclude_diag=True
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    if ground_truth is not None:
        if query_indices is not None:
            # Compute recall only on sampled queries
            recall = compute_recall(ground_truth, indices[query_indices])
        else:
            recall = compute_recall(ground_truth, indices)
    else:
        recall = 1.0

    return elapsed, recall, mem_mb


def generate_configs(n_samples):
    """Generate benchmark configs based on dataset size."""
    configs = []

    # nlist and nprobe ranges following FAISS guidelines
    # FAISS recommends: 4*sqrt(n) to 16*sqrt(n) for nlist
    # For 10M: sqrt(10M) = 3162, so nlist in range 12k-50k
    if n_samples <= 100_000:
        nlist_values = [256, 1024]
        nprobe_ratios = [0.01, 0.05, 0.1, 0.25, 0.5]
    elif n_samples <= 1_000_000:
        nlist_values = [1024, 4096]
        nprobe_ratios = [0.01, 0.05, 0.1, 0.25, 0.5]
    else:
        # 10M+ samples: nlist=16384 or 65536
        nlist_values = [16384, 65536]
        nprobe_ratios = [0.005, 0.01, 0.025, 0.05, 0.1]

    for nlist in nlist_values:
        for ratio in nprobe_ratios:
            nprobe = max(1, int(nlist * ratio))

            # IVF
            configs.append(
                {
                    "name": f"IVF-nlist{nlist}-np{nprobe}",
                    "config": FaissConfig(index_type="IVF", nlist=nlist, nprobe=nprobe),
                    "index_type": "IVF",
                    "nlist": nlist,
                    "nprobe": nprobe,
                    "nprobe_ratio": ratio,
                }
            )

            # IVFPQ with M=16
            configs.append(
                {
                    "name": f"IVFPQ-M16-nlist{nlist}-np{nprobe}",
                    "config": FaissConfig(
                        index_type="IVFPQ", nlist=nlist, M=16, nprobe=nprobe
                    ),
                    "index_type": "IVFPQ",
                    "nlist": nlist,
                    "nprobe": nprobe,
                    "nprobe_ratio": ratio,
                }
            )

    return configs


def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS index types")
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--n_features", type=int, default=128)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument(
        "--data",
        type=str,
        choices=["clustered", "random"],
        default="clustered",
        help="Data type: 'clustered' (realistic) or 'random'",
    )
    args = parser.parse_args()

    print(f"FAISS Benchmark | {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.n_samples:,} x {args.n_features}D | k={args.k}")
    print("=" * 70)

    # Generate data
    torch.manual_seed(args.seed)

    if args.data == "clustered":
        # Gaussian mixture for realistic IVFPQ performance
        n_clusters = min(1000, args.n_samples // 100)
        samples_per_cluster = args.n_samples // n_clusters
        centers = torch.randn(n_clusters, args.n_features) * 10

        X_list = []
        for i in range(n_clusters):
            n_pts = (
                samples_per_cluster
                if i < n_clusters - 1
                else args.n_samples - len(X_list) * samples_per_cluster
            )
            points = centers[i] + torch.randn(n_pts, args.n_features) * 0.5
            X_list.append(points)

        X = torch.cat(X_list, dim=0).cuda()
        print(f"Data: {n_clusters} Gaussian clusters")
    else:
        X = torch.randn(args.n_samples, args.n_features).cuda()
        print("Data: random (uniform)")

    # Ground truth from Flat index
    # For large datasets (>1M), use sampled queries and skip Flat timing
    n_query_samples = min(10000, args.n_samples)
    use_sampled = args.n_samples > 1_000_000

    if use_sampled:
        print(f"\nUsing {n_query_samples:,} sampled queries for recall evaluation")
        query_indices = torch.randperm(args.n_samples)[:n_query_samples]
        query_indices, _ = query_indices.sort()  # Keep sorted for efficiency

        print(
            "Computing ground truth (FAISS Flat on sampled queries)...",
            end=" ",
            flush=True,
        )

        # Use FAISS GPU directly for fast ground truth computation
        import faiss

        X_np = X.cpu().numpy().astype(np.float32)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, args.n_features)
        index.add(X_np)

        query_np = X_np[query_indices.cpu().numpy()]
        _, ground_truth_np = index.search(query_np, args.k + 1)
        # Remove self from results (exclude_diag)
        ground_truth = torch.from_numpy(ground_truth_np[:, 1:]).cuda()
        print("done")

        # No Flat timing for large datasets (too slow)
        flat_time = None
        flat_mem = 0
    else:
        query_indices = None
        print("\nComputing ground truth (Flat)...", end=" ", flush=True)
        flat_config = FaissConfig(index_type="Flat")
        flat_time, _, flat_mem = benchmark_config(X, flat_config, args.k)
        _, ground_truth = pairwise_distances(
            X, k=args.k, backend=flat_config, return_indices=True, exclude_diag=True
        )
        print(f"{flat_time:.2f}s")

    # Results list
    results = []
    if flat_time is not None:
        results.append(
            {
                "name": "Flat",
                "index_type": "Flat",
                "nlist": 0,
                "nprobe": 0,
                "nprobe_ratio": 0,
                "time_s": flat_time,
                "recall": 1.0,
                "memory_mb": flat_mem,
                "speedup": 1.0,
            }
        )

    # Benchmark configs
    configs = generate_configs(args.n_samples)
    print(f"\nBenchmarking {len(configs)} configurations...")
    print(f"\n{'Config':<30} {'Time':>8} {'Speedup':>8} {'Recall':>8}")
    print("-" * 60)
    if flat_time is not None:
        print(f"{'Flat (exact)':<30} {flat_time:>7.2f}s {1.0:>7.1f}x {1.0:>7.1%}")

    for cfg in configs:
        torch.cuda.empty_cache()
        try:
            elapsed, recall, mem_mb = benchmark_config(
                X, cfg["config"], args.k, ground_truth, query_indices
            )
            speedup = flat_time / elapsed if flat_time else 0.0

            results.append(
                {
                    "name": cfg["name"],
                    "index_type": cfg["index_type"],
                    "nlist": cfg["nlist"],
                    "nprobe": cfg["nprobe"],
                    "nprobe_ratio": cfg["nprobe_ratio"],
                    "time_s": elapsed,
                    "recall": recall,
                    "memory_mb": mem_mb,
                    "speedup": speedup,
                }
            )

            if flat_time:
                print(
                    f"{cfg['name']:<30} {elapsed:>7.2f}s {speedup:>7.1f}x {recall:>7.1%}"
                )
            else:
                print(f"{cfg['name']:<30} {elapsed:>7.2f}s {'N/A':>8} {recall:>7.1%}")

        except Exception as e:
            print(f"{cfg['name']:<30} FAILED: {str(e)[:40]}")

    # Save results
    print(f"\nSaving results to {args.output}")
    fieldnames = [
        "name",
        "index_type",
        "nlist",
        "nprobe",
        "nprobe_ratio",
        "time_s",
        "recall",
        "memory_mb",
        "speedup",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    approx = [r for r in results if r["index_type"] != "Flat"]
    for threshold in [0.95, 0.90, 0.80]:
        candidates = [r for r in approx if r["recall"] >= threshold]
        if candidates:
            best = min(candidates, key=lambda x: x["time_s"])
            print(f"  Fastest with >{threshold:.0%} recall: {best['name']}")
            print(
                f"    -> {best['time_s']:.2f}s, {best['recall']:.1%} recall, {best['speedup']:.1f}x speedup"
            )


if __name__ == "__main__":
    main()
