"""Benchmark FAISS index types (Flat, IVF, IVFPQ) on single and multi-GPU.

This benchmark compares k-NN computation performance across different FAISS
index types for large-scale datasets.
"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchdr.distance import pairwise_distances, FaissConfig


def generate_synthetic_data(
    n_samples: int, n_features: int, seed: int = 42
) -> torch.Tensor:
    """Generate synthetic data for benchmarking."""
    torch.manual_seed(seed)
    return torch.randn(n_samples, n_features)


def benchmark_faiss_config(
    X: torch.Tensor,
    config: FaissConfig,
    k: int = 15,
    n_warmup: int = 1,
    n_runs: int = 3,
    use_dataloader: bool = False,
    batch_size: int = 10000,
) -> Dict:
    """Benchmark a single FAISS configuration.

    Parameters
    ----------
    X : torch.Tensor
        Input data.
    config : FaissConfig
        FAISS configuration to benchmark.
    k : int
        Number of nearest neighbors.
    n_warmup : int
        Number of warmup runs (not timed).
    n_runs : int
        Number of timed runs.
    use_dataloader : bool
        If True, use DataLoader input instead of tensor.
    batch_size : int
        Batch size for DataLoader.

    Returns
    -------
    results : dict
        Dictionary with timing results and metadata.
    """
    if use_dataloader:
        dataset = TensorDataset(X)
        data_input = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        data_input = X

    # Warmup runs
    for _ in range(n_warmup):
        if use_dataloader:
            _ = pairwise_distances(data_input, k=k, backend=config, return_indices=True)
        else:
            _ = pairwise_distances(data_input, k=k, backend=config, return_indices=True)

    # Synchronize GPU before timing
    if X.is_cuda:
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        dist, idx = pairwise_distances(
            data_input, k=k, backend=config, return_indices=True
        )
        if X.is_cuda or (hasattr(dist, "is_cuda") and dist.is_cuda):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "config": repr(config),
        "index_type": config.index_type,
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "k": k,
    }


def compute_recall(
    X: torch.Tensor,
    config: FaissConfig,
    k: int = 15,
) -> float:
    """Compute recall of approximate index vs exact Flat index."""
    # Get exact results
    dist_exact, idx_exact = pairwise_distances(
        X, k=k, backend="faiss", return_indices=True
    )

    # Get approximate results
    dist_approx, idx_approx = pairwise_distances(
        X, k=k, backend=config, return_indices=True
    )

    # Compute recall
    recall = 0
    for i in range(len(X)):
        true_neighbors = set(idx_exact[i].tolist())
        found_neighbors = set(idx_approx[i].tolist())
        recall += len(true_neighbors & found_neighbors) / k
    recall /= len(X)

    return recall


def run_benchmark(
    dataset_sizes: List[int],
    n_features: int = 128,
    k: int = 15,
    device: str = "cuda",
    n_runs: int = 3,
    compute_recall_flag: bool = True,
) -> List[Dict]:
    """Run full benchmark suite.

    Parameters
    ----------
    dataset_sizes : list of int
        Dataset sizes to benchmark.
    n_features : int
        Number of features (must be divisible by M for IVFPQ).
    k : int
        Number of nearest neighbors.
    device : str
        Device to run on ("cuda" or "cpu").
    n_runs : int
        Number of timed runs per configuration.
    compute_recall_flag : bool
        Whether to compute recall for approximate indexes.

    Returns
    -------
    results : list of dict
        Benchmark results for all configurations.
    """
    results = []

    # Define configurations to benchmark
    configs = {
        "Flat": FaissConfig(index_type="Flat"),
        "IVF (nprobe=1)": FaissConfig(index_type="IVF", nprobe=1),
        "IVF (nprobe=10)": FaissConfig(index_type="IVF", nprobe=10),
        "IVF (nprobe=50)": FaissConfig(index_type="IVF", nprobe=50),
        "IVFPQ (nprobe=1, M=16)": FaissConfig(
            index_type="IVFPQ", nprobe=1, M=16, nbits=8
        ),
        "IVFPQ (nprobe=10, M=16)": FaissConfig(
            index_type="IVFPQ", nprobe=10, M=16, nbits=8
        ),
        "IVFPQ (nprobe=50, M=16)": FaissConfig(
            index_type="IVFPQ", nprobe=50, M=16, nbits=8
        ),
        "IVFPQ (nprobe=10, M=32)": FaissConfig(
            index_type="IVFPQ", nprobe=10, M=32, nbits=8
        ),
    }

    for n_samples in dataset_sizes:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {n_samples:,} samples x {n_features} features")
        print(f"{'=' * 60}")

        # Generate data
        X = generate_synthetic_data(n_samples, n_features)
        if device == "cuda":
            X = X.cuda()

        for config_name, config in configs.items():
            print(f"\n  {config_name}...", end=" ", flush=True)

            try:
                result = benchmark_faiss_config(
                    X, config, k=k, n_runs=n_runs, n_warmup=1
                )
                result["config_name"] = config_name

                # Compute recall for approximate indexes
                if compute_recall_flag and config.index_type != "Flat":
                    # Use smaller subset for recall computation on large datasets
                    if n_samples > 50000:
                        X_recall = X[:50000]
                    else:
                        X_recall = X
                    result["recall"] = compute_recall(X_recall, config, k=k)
                    print(
                        f"{result['mean_time']:.3f}s (±{result['std_time']:.3f}s), recall={result['recall']:.2%}"
                    )
                else:
                    result["recall"] = 1.0 if config.index_type == "Flat" else None
                    print(f"{result['mean_time']:.3f}s (±{result['std_time']:.3f}s)")

                results.append(result)

            except Exception as e:
                print(f"FAILED: {e}")
                results.append(
                    {
                        "config_name": config_name,
                        "n_samples": n_samples,
                        "error": str(e),
                    }
                )

        # Clear GPU memory
        del X
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def print_summary_table(results: List[Dict]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    # Group by dataset size
    sizes = sorted(set(r.get("n_samples", 0) for r in results if "error" not in r))

    for n_samples in sizes:
        print(f"\n{n_samples:,} samples:")
        print("-" * 80)
        print(f"{'Config':<35} {'Time (s)':<15} {'Speedup':<12} {'Recall':<10}")
        print("-" * 80)

        size_results = [
            r for r in results if r.get("n_samples") == n_samples and "error" not in r
        ]
        if not size_results:
            continue

        # Find Flat baseline
        flat_time = next(
            (r["mean_time"] for r in size_results if r["index_type"] == "Flat"), None
        )

        for r in size_results:
            speedup = flat_time / r["mean_time"] if flat_time else 0
            recall_str = f"{r['recall']:.2%}" if r.get("recall") is not None else "N/A"
            print(
                f"{r['config_name']:<35} {r['mean_time']:<15.3f} {speedup:<12.2f}x {recall_str:<10}"
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS index types")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100000, 500000, 1000000],
        help="Dataset sizes to benchmark",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=128,
        help="Number of features (must be divisible by M for IVFPQ)",
    )
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument(
        "--no-recall",
        action="store_true",
        help="Skip recall computation",
    )
    args = parser.parse_args()

    print("FAISS Index Benchmark")
    print("=" * 60)
    print(f"Device: {args.device}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset sizes: {args.sizes}")
    print(f"Features: {args.features}")
    print(f"k: {args.k}")
    print(f"Runs per config: {args.runs}")

    results = run_benchmark(
        dataset_sizes=args.sizes,
        n_features=args.features,
        k=args.k,
        device=args.device,
        n_runs=args.runs,
        compute_recall_flag=not args.no_recall,
    )

    print_summary_table(results)


if __name__ == "__main__":
    main()
