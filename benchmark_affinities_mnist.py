#!/usr/bin/env python3
"""Benchmark runtime and memory usage for various affinities on full MNIST dataset."""

import torch
import numpy as np
import time
import tracemalloc
import gc
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import all affinity types
from torchdr.affinity import (
    GaussianAffinity,
    StudentAffinity,
    EntropicAffinity,
    SymmetricEntropicAffinity,
    SinkhornAffinity,
    DoublyStochasticQuadraticAffinity,
    UMAPAffinity,
    SelfTuningAffinity,
    MAGICAffinity,
    PACMAPAffinity,
)


def load_mnist_full(n_samples=70000):
    """Load full MNIST dataset."""
    print(f"Loading MNIST data (n={n_samples})...")

    # Fetch MNIST
    mnist = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # Take subset if specified
    if n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    # Normalize
    X = X.astype(np.float32) / 255.0

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X.astype(np.float32), y


def benchmark_affinity(affinity_class, X, affinity_name, **kwargs):
    """Benchmark a single affinity type."""
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {affinity_name}")
    print(f"{'=' * 60}")

    # Test different precision modes if supported
    precisions = ["32-true"]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        precisions.extend(["16-mixed", "bf16-mixed"])

    for precision in precisions:
        print(f"\n  Precision: {precision}")
        print(f"  {'-' * 40}")

        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

            # Create affinity with current precision
            affinity_kwargs = kwargs.copy()
            affinity_kwargs["device"] = device
            affinity_kwargs["verbose"] = False
            affinity_kwargs["precision"] = precision

            affinity = affinity_class(**affinity_kwargs)

            # Memory measurement
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                tracemalloc.start()
                mem_before = 0

            # Warmup run (small batch)
            if device == "cuda":
                X_warmup = X[:100]
                _ = affinity(X_warmup)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Timing runs
            n_runs = 3 if X.shape[0] <= 10000 else 1
            times = []

            for run in range(n_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                # Compute affinity matrix
                affinity_matrix = affinity(X)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append(end_time - start_time)

                # Get memory usage from first run
                if run == 0:
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                        mem_after = torch.cuda.memory_allocated() / 1024 / 1024
                    else:
                        current, peak = tracemalloc.get_traced_memory()
                        peak_memory = peak / 1024 / 1024
                        mem_after = current / 1024 / 1024
                        tracemalloc.stop()

            # Calculate metrics
            runtime = np.mean(times)
            runtime_std = np.std(times) if len(times) > 1 else 0
            memory_used = peak_memory - mem_before

            # Get output shape and sparsity
            if hasattr(affinity_matrix, "shape"):
                output_shape = str(affinity_matrix.shape)
                if hasattr(affinity_matrix, "nnz"):  # Sparse matrix
                    sparsity = 1.0 - (
                        affinity_matrix.nnz
                        / (affinity_matrix.shape[0] * affinity_matrix.shape[1])
                    )
                    output_type = "sparse"
                else:
                    sparsity = 0.0
                    output_type = "dense"
            else:
                output_shape = "LazyTensor"
                sparsity = 0.0
                output_type = "lazy"

            # Store results
            result_key = f"{affinity_name}_{precision}"
            results[result_key] = {
                "Affinity": affinity_name,
                "Precision": precision,
                "Runtime (s)": runtime,
                "Runtime Std (s)": runtime_std,
                "Peak Memory (MB)": memory_used,
                "Total Peak Memory (MB)": peak_memory,
                "Output Shape": output_shape,
                "Output Type": output_type,
                "Sparsity": sparsity,
                "Status": "Success",
            }

            print("  ✓ Success!")
            print(f"  Runtime: {runtime:.2f}s ± {runtime_std:.2f}s")
            print(f"  Peak Memory: {memory_used:.2f} MB")
            print(f"  Output: {output_type} {output_shape}")
            if sparsity > 0:
                print(f"  Sparsity: {sparsity:.1%}")

            # Clear affinity to free memory
            del affinity
            if "affinity_matrix" in locals():
                del affinity_matrix

        except Exception as e:
            result_key = f"{affinity_name}_{precision}"
            results[result_key] = {
                "Affinity": affinity_name,
                "Precision": precision,
                "Runtime (s)": np.nan,
                "Runtime Std (s)": np.nan,
                "Peak Memory (MB)": np.nan,
                "Total Peak Memory (MB)": np.nan,
                "Output Shape": "N/A",
                "Output Type": "N/A",
                "Sparsity": np.nan,
                "Status": f"Failed: {str(e)[:100]}",
            }
            print(f"  ✗ Failed: {str(e)[:150]}")

        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return results


def generate_report(df, n_samples, output_prefix="affinity_benchmark"):
    """Generate comprehensive report with visualizations."""

    report = []
    report.append("=" * 80)
    report.append("AFFINITY BENCHMARK REPORT - RUNTIME AND MEMORY ANALYSIS")
    report.append("=" * 80)
    report.append(f"\nDataset: MNIST ({n_samples} samples, 784 features)")
    report.append(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        report.append(f"GPU: {torch.cuda.get_device_name()}")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n")

    # Filter successful runs
    successful = df[df["Status"] == "Success"].copy()

    if successful.empty:
        report.append("No successful runs to analyze.")
        return "\n".join(report)

    # Summary by affinity type
    report.append("-" * 80)
    report.append("SUMMARY BY AFFINITY TYPE")
    report.append("-" * 80)

    for affinity in successful["Affinity"].unique():
        affinity_data = successful[successful["Affinity"] == affinity]
        baseline = affinity_data[affinity_data["Precision"] == "32-true"]

        if baseline.empty:
            continue

        baseline = baseline.iloc[0]
        report.append(f"\n{affinity}:")
        report.append(f"  Output Type: {baseline['Output Type']}")
        report.append("  Baseline (32-true):")
        report.append(f"    Runtime: {baseline['Runtime (s)']:.2f}s")
        report.append(f"    Peak Memory: {baseline['Peak Memory (MB)']:.2f} MB")
        if baseline["Sparsity"] > 0:
            report.append(f"    Sparsity: {baseline['Sparsity']:.1%}")

        for _, row in affinity_data.iterrows():
            if row["Precision"] != "32-true":
                runtime_speedup = baseline["Runtime (s)"] / row["Runtime (s)"]
                memory_reduction = (
                    (baseline["Peak Memory (MB)"] - row["Peak Memory (MB)"])
                    / baseline["Peak Memory (MB)"]
                    * 100
                )

                report.append(f"\n  {row['Precision']}:")
                report.append(
                    f"    Runtime: {row['Runtime (s)']:.2f}s (Speedup: {runtime_speedup:.2f}x)"
                )
                report.append(
                    f"    Peak Memory: {row['Peak Memory (MB)']:.2f} MB (Reduction: {memory_reduction:.1f}%)"
                )

    # Create visualizations
    if len(successful) > 0:
        # Setup plot style
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.figure(figsize=(20, 12))

        # 1. Runtime comparison
        ax1 = plt.subplot(2, 3, 1)
        runtime_pivot = successful.pivot(
            index="Affinity", columns="Precision", values="Runtime (s)"
        )
        runtime_pivot.plot(kind="bar", ax=ax1)
        ax1.set_title(
            "Runtime Comparison by Precision Mode", fontsize=12, fontweight="bold"
        )
        ax1.set_ylabel("Runtime (seconds)", fontsize=10)
        ax1.set_xlabel("Affinity Type", fontsize=10)
        ax1.legend(title="Precision", fontsize=9)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 2. Memory comparison
        ax2 = plt.subplot(2, 3, 2)
        memory_pivot = successful.pivot(
            index="Affinity", columns="Precision", values="Peak Memory (MB)"
        )
        memory_pivot.plot(kind="bar", ax=ax2)
        ax2.set_title(
            "Peak Memory Comparison by Precision Mode", fontsize=12, fontweight="bold"
        )
        ax2.set_ylabel("Peak Memory (MB)", fontsize=10)
        ax2.set_xlabel("Affinity Type", fontsize=10)
        ax2.legend(title="Precision", fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 3. Runtime vs Memory scatter plot
        ax3 = plt.subplot(2, 3, 3)
        for precision in successful["Precision"].unique():
            data = successful[successful["Precision"] == precision]
            ax3.scatter(
                data["Runtime (s)"],
                data["Peak Memory (MB)"],
                label=precision,
                s=100,
                alpha=0.7,
            )
        ax3.set_xlabel("Runtime (seconds)", fontsize=10)
        ax3.set_ylabel("Peak Memory (MB)", fontsize=10)
        ax3.set_title("Runtime vs Memory Trade-off", fontsize=12, fontweight="bold")
        ax3.legend(title="Precision", fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. Speedup heatmap (if mixed precision data exists)
        ax4 = plt.subplot(2, 3, 4)
        speedup_data = []
        for affinity in successful["Affinity"].unique():
            affinity_data = successful[successful["Affinity"] == affinity]
            baseline = affinity_data[affinity_data["Precision"] == "32-true"]
            if baseline.empty:
                continue
            baseline_runtime = baseline.iloc[0]["Runtime (s)"]

            for precision in ["16-mixed", "bf16-mixed"]:
                precision_data = affinity_data[affinity_data["Precision"] == precision]
                if not precision_data.empty:
                    speedup = baseline_runtime / precision_data.iloc[0]["Runtime (s)"]
                    speedup_data.append(
                        {
                            "Affinity": affinity,
                            "Precision": precision,
                            "Speedup": speedup,
                        }
                    )

        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            speedup_pivot = speedup_df.pivot(
                index="Affinity", columns="Precision", values="Speedup"
            )
            sns.heatmap(
                speedup_pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=1.0,
                ax=ax4,
                cbar_kws={"label": "Speedup Factor"},
            )
            ax4.set_title(
                "Runtime Speedup with Mixed Precision", fontsize=12, fontweight="bold"
            )
            ax4.set_xlabel("Precision Mode", fontsize=10)
            ax4.set_ylabel("Affinity Type", fontsize=10)

        # 5. Memory reduction heatmap
        ax5 = plt.subplot(2, 3, 5)
        memory_reduction_data = []
        for affinity in successful["Affinity"].unique():
            affinity_data = successful[successful["Affinity"] == affinity]
            baseline = affinity_data[affinity_data["Precision"] == "32-true"]
            if baseline.empty:
                continue
            baseline_memory = baseline.iloc[0]["Peak Memory (MB)"]

            for precision in ["16-mixed", "bf16-mixed"]:
                precision_data = affinity_data[affinity_data["Precision"] == precision]
                if not precision_data.empty:
                    reduction = (
                        (baseline_memory - precision_data.iloc[0]["Peak Memory (MB)"])
                        / baseline_memory
                        * 100
                    )
                    memory_reduction_data.append(
                        {
                            "Affinity": affinity,
                            "Precision": precision,
                            "Memory Reduction (%)": reduction,
                        }
                    )

        if memory_reduction_data:
            memory_reduction_df = pd.DataFrame(memory_reduction_data)
            memory_pivot = memory_reduction_df.pivot(
                index="Affinity", columns="Precision", values="Memory Reduction (%)"
            )
            sns.heatmap(
                memory_pivot,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                ax=ax5,
                cbar_kws={"label": "Memory Reduction (%)"},
            )
            ax5.set_title(
                "Memory Reduction with Mixed Precision", fontsize=12, fontweight="bold"
            )
            ax5.set_xlabel("Precision Mode", fontsize=10)
            ax5.set_ylabel("Affinity Type", fontsize=10)

        # 6. Affinity type comparison (sorted by runtime)
        ax6 = plt.subplot(2, 3, 6)
        baseline_data = successful[successful["Precision"] == "32-true"].copy()
        baseline_data = baseline_data.sort_values("Runtime (s)")
        colors = [
            "green" if x == "sparse" else "blue" if x == "dense" else "orange"
            for x in baseline_data["Output Type"]
        ]
        ax6.barh(range(len(baseline_data)), baseline_data["Runtime (s)"], color=colors)
        ax6.set_yticks(range(len(baseline_data)))
        ax6.set_yticklabels(baseline_data["Affinity"])
        ax6.set_xlabel("Runtime (seconds)", fontsize=10)
        ax6.set_title(
            "Affinity Types Ranked by Runtime (32-bit)", fontsize=12, fontweight="bold"
        )
        ax6.grid(True, alpha=0.3, axis="x")

        # Add legend for output types
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="green", label="Sparse"),
            Patch(facecolor="blue", label="Dense"),
            Patch(facecolor="orange", label="Lazy"),
        ]
        ax6.legend(handles=legend_elements, loc="lower right", title="Output Type")

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_comparison.png", dpi=150, bbox_inches="tight")
        print(f"\nVisualization saved to {output_prefix}_comparison.png")

    # Ranking table
    report.append("\n" + "-" * 80)
    report.append("PERFORMANCE RANKING (32-bit precision)")
    report.append("-" * 80)

    baseline_data = successful[successful["Precision"] == "32-true"].copy()
    baseline_data = baseline_data.sort_values("Runtime (s)")

    report.append("\nFastest Affinities:")
    for i, (_, row) in enumerate(baseline_data.head(5).iterrows(), 1):
        report.append(
            f"  {i}. {row['Affinity']}: {row['Runtime (s)']:.2f}s, {row['Peak Memory (MB)']:.0f}MB ({row['Output Type']})"
        )

    report.append("\nMost Memory Efficient:")
    memory_sorted = baseline_data.sort_values("Peak Memory (MB)")
    for i, (_, row) in enumerate(memory_sorted.head(5).iterrows(), 1):
        report.append(
            f"  {i}. {row['Affinity']}: {row['Peak Memory (MB)']:.0f}MB, {row['Runtime (s)']:.2f}s ({row['Output Type']})"
        )

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """Run affinity benchmarks on MNIST."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("Affinity Benchmark on Full MNIST Dataset")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"BF16 Support: {torch.cuda.is_bf16_supported()}")
    print()

    # Load MNIST - adjust sample size as needed
    n_samples = 10000  # Start with 10k, can increase to 70000 for full dataset
    X, y = load_mnist_full(n_samples)
    print(f"Data shape: {X.shape}")
    print(f"Data dtype: {X.dtype}")
    print(f"Memory footprint: {X.nbytes / 1024 / 1024:.2f} MB")

    # Define affinities to test
    affinities_to_test = [
        # Basic affinities
        (GaussianAffinity, "Gaussian", {"sigma": 1.0}),
        (StudentAffinity, "Student-t", {}),
        # Entropic affinities
        (EntropicAffinity, "Entropic", {"perplexity": 30}),
        (SymmetricEntropicAffinity, "SymmetricEntropic", {"perplexity": 30}),
        # Doubly stochastic affinities
        (SinkhornAffinity, "Sinkhorn", {"eps": 0.1, "max_iter": 100}),
        (DoublyStochasticQuadraticAffinity, "DoublyStochasticQuad", {"max_iter": 10}),
        # Method-specific affinities
        (UMAPAffinity, "UMAP", {"n_neighbors": 30}),
        (SelfTuningAffinity, "SelfTuning", {"perplexity": 30}),
        (MAGICAffinity, "MAGIC", {"knn": 30}),
        (PACMAPAffinity, "PACMAP", {"n_neighbors": 30}),
        # (PHATEAffinity, "PHATE", {"knn": 30}),  # May be slow on large datasets
    ]

    # Run benchmarks
    all_results = []
    for affinity_class, affinity_name, kwargs in affinities_to_test:
        results = benchmark_affinity(affinity_class, X, affinity_name, **kwargs)
        all_results.extend(results.values())

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"affinity_benchmark_results_{n_samples}samples_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nRaw results saved to {csv_file}")

    # Generate and print report
    report = generate_report(
        df,
        n_samples,
        output_prefix=f"affinity_benchmark_{n_samples}samples_{timestamp}",
    )
    print("\n" + report)

    # Save report to file
    report_file = f"affinity_benchmark_report_{n_samples}samples_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
