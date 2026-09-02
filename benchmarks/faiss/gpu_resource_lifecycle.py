"""Benchmark repeated exact k-NN searches on one GPU.

Measures what FAISS GPU resource ownership costs: the first-call initialization,
the steady-state search time of the calls that follow, the peak memory the
device holds while they run, and how much stays resident afterwards. FAISS
allocates its temporary pool outside the PyTorch caching allocator, so device
occupancy is read from ``torch.cuda.mem_get_info`` rather than from
``max_memory_allocated``.

Three modes cover how TorchDR reaches FAISS in practice:

- ``fresh-config``: a new ``FaissConfig`` per call, which is what every
  ``backend="faiss"`` estimator does.
- ``shared-config``: one ``FaissConfig`` reused across calls.
- ``affinity``: repeated ``UMAPAffinity`` fits, an end-to-end path that adds
  normalization and symmetrization on top of the search.

Run the same command on the base commit and on the branch to compare. Each mode
prints and persists its row as soon as it finishes, so a failure in a later mode
does not discard earlier results. The run also prints the resolved ``torchdr``
package path, because an editable install can shadow the checkout under test.

Usage:
    PYTHONPATH=$(git rev-parse --show-toplevel) python gpu_resource_lifecycle.py \
        --n 200000 --d 128 --k 15 --repeats 20 --output results.json
"""

import argparse
import json
import resource
import statistics
import time

import torch

import torchdr
from torchdr.affinity import UMAPAffinity
from torchdr.distance import FaissConfig, pairwise_distances

MODES = ("fresh-config", "shared-config", "affinity")
ROW = "{:<14}{:>10}{:>10}{:>10}{:>10}{:>9}{:>9}{:>9}"


def device_used_mb() -> float:
    """Megabytes occupied on the current CUDA device, including FAISS' own pools."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def host_peak_mb() -> float:
    """Peak resident set size of this process, in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_mode(name, call, repeats):
    """Time ``call`` ``repeats`` times and return its summary and last result.

    Occupancy is sampled after every call, then once more with the result
    dropped and the PyTorch cache released, which leaves only what FAISS still
    holds on the device.
    """
    torch.cuda.synchronize()
    durations = []
    peak_mb = 0.0
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        peak_mb = max(peak_mb, device_used_mb())

    scratch, result = result, None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    steady = durations[1:] or durations
    return {
        "mode": name,
        "calls": len(durations),
        "first_call_s": durations[0],
        "steady_median_s": statistics.median(steady),
        "steady_mean_s": statistics.fmean(steady),
        "steady_max_s": max(steady),
        "total_s": sum(durations),
        "peak_device_mb": peak_mb,
        "resident_device_mb": device_used_mb(),
        "peak_host_mb": host_peak_mb(),
    }, scratch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000, help="database size")
    parser.add_argument("--d", type=int, default=128, help="dimension")
    parser.add_argument("--k", type=int, default=15, help="neighbors per query")
    parser.add_argument("--repeats", type=int, default=20, help="calls per mode")
    parser.add_argument(
        "--affinity-repeats", type=int, default=5, help="UMAPAffinity fits"
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", type=str, default="", help="provenance label")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")

    torch.manual_seed(args.seed)
    X = torch.randn(args.n, args.d, device="cuda")
    torch.cuda.synchronize()

    report = {
        "n": args.n,
        "d": args.d,
        "k": args.k,
        "repeats": args.repeats,
        "seed": args.seed,
        "label": args.label,
        "gpu": torch.cuda.get_device_name(0),
        "torchdr_path": torchdr.__file__,
        "data_only_device_mb": device_used_mb(),
        "modes": [],
    }

    print(f"n={args.n} d={args.d} k={args.k} repeats={args.repeats} seed={args.seed}")
    print(f"label={args.label or '-'} gpu={report['gpu']}")
    print(f"torchdr={report['torchdr_path']}")
    print(f"data_only_device_mb={report['data_only_device_mb']:.0f}")
    print(
        ROW.format(
            "mode", "first", "median", "mean", "total", "peakMB", "residMB", "hostMB"
        )
    )

    calls = {
        "fresh-config": (
            lambda: pairwise_distances(
                X,
                k=args.k,
                backend=FaissConfig(),
                return_indices=True,
                exclude_diag=True,
            ),
            args.repeats,
        ),
        "shared-config": (
            lambda config=FaissConfig(): pairwise_distances(
                X, k=args.k, backend=config, return_indices=True, exclude_diag=True
            ),
            args.repeats,
        ),
        "affinity": (
            lambda: UMAPAffinity(n_neighbors=args.k, backend="faiss", verbose=False)(X),
            args.affinity_repeats,
        ),
    }

    neighbors = {}
    for name in args.modes:
        call, repeats = calls[name]
        summary, result = run_mode(name, call, repeats)
        report["modes"].append(summary)
        print(
            ROW.format(
                summary["mode"],
                f"{summary['first_call_s']:.4f}",
                f"{summary['steady_median_s']:.4f}",
                f"{summary['steady_mean_s']:.4f}",
                f"{summary['total_s']:.4f}",
                f"{summary['peak_device_mb']:.0f}",
                f"{summary['resident_device_mb']:.0f}",
                f"{summary['peak_host_mb']:.0f}",
            )
        )
        if name in ("fresh-config", "shared-config"):
            neighbors[name] = result[1]
        if args.output:
            with open(args.output, "w") as handle:
                json.dump(report, handle, indent=2)

    if len(neighbors) == 2:
        agree = torch.equal(*neighbors.values())
        report["indices_agree_across_modes"] = agree
        print(f"indices agree across modes: {agree}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
