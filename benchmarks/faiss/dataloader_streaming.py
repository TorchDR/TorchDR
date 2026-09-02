"""Benchmark k-NN over a DataLoader, from batch staging to final neighbors.

The DataLoader path streams host batches into a FAISS index, so what it costs
is decided as much by how a batch reaches the index as by the search itself.
This measures the whole path: the build pass that trains and adds every batch,
the query pass that searches them back, the peak resident set of the process,
and the peak occupancy of the device. FAISS allocates outside the PyTorch
caching allocator, so device memory is read from ``torch.cuda.mem_get_info``
rather than from ``max_memory_allocated``.

Timing and memory are collected in separate passes: the memory sampler polls
from another thread and would otherwise perturb the numbers it sits next to.

Run the same command on the base commit and on the branch to compare NumPy
staging with tensor-native staging. Each row is printed and persisted as soon
as it finishes, so a failure in a later case does not discard earlier results.
The run also prints the resolved ``torchdr`` package path, because an editable
install can shadow the checkout under test.

Usage:
    PYTHONPATH=$(git rev-parse --show-toplevel) python dataloader_streaming.py \
        --n 500000 --d 128 --batch-sizes 1024 8192 65536 --output results.json
"""

import argparse
import inspect
import json
import os
import statistics
import threading
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchdr
from torchdr.distance import FaissConfig, pairwise_distances
from torchdr.distance import faiss as faiss_module

ROW = "{:<12}{:>8}{:>16}{:>9}{:>9}{:>9}{:>9}{:>9}{:>7}"
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def device_used_mb() -> float:
    """Megabytes occupied on the current CUDA device, including FAISS' own pools."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def host_rss_mb() -> float:
    """Resident set size of this process right now, in megabytes."""
    with open("/proc/self/statm") as handle:
        pages = int(handle.read().split()[1])
    return pages * PAGE_SIZE / 1024**2


class MemorySampler:
    """Poll host and device occupancy while a call runs."""

    def __init__(self, device: int, interval: float = 0.01):
        self.device = device
        self.interval = interval
        self.host_mb = 0.0
        self.device_mb = 0.0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        torch.cuda.set_device(self.device)
        while not self._done.is_set():
            self.host_mb = max(self.host_mb, host_rss_mb())
            self.device_mb = max(self.device_mb, device_used_mb())
            self._done.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._done.set()
        self._thread.join()


def instrument():
    """Time the build and query passes wherever the module defines them.

    The two passes are private helpers whose names and signatures differ across
    the revisions being compared, so they are wrapped by name and called
    through, rather than invoked directly.
    """
    totals = {"build_s": 0.0, "search_s": 0.0}
    originals = {}

    def wrap(name, key):
        original = getattr(faiss_module, name, None)
        if original is None:
            return
        originals[name] = original

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            totals[key] += time.perf_counter() - start
            return result

        setattr(faiss_module, name, timed)

    wrap("_build_index_from_dataloader", "build_s")
    wrap("_search_all_from_dataloader", "search_s")
    wrap("_search_from_dataloader", "search_s")
    return totals, originals


def restore(originals):
    """Undo :func:`instrument`."""
    for name, original in originals.items():
        setattr(faiss_module, name, original)


def timed_call(call, repeats):
    """Run ``call`` and return per-pass timings, the best total, and the result."""
    totals, originals = instrument()
    try:
        durations = []
        phases = []
        result = None
        for _ in range(repeats):
            before = dict(totals)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = call()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            phases.append({k: totals[k] - before[k] for k in totals})
    finally:
        restore(originals)

    median = statistics.median(durations)
    closest = min(phases, key=lambda p: abs(sum(p.values()) - median))
    return {
        "total_s": median,
        "total_min_s": min(durations),
        "build_s": closest["build_s"],
        "search_s": closest["search_s"],
        "repeats": repeats,
    }, result


def sampled_call(call, device):
    """Run ``call`` once while sampling occupancy, and return the peaks."""
    torch.cuda.synchronize()
    sampler = MemorySampler(device)
    sampler.start()
    try:
        call()
    finally:
        torch.cuda.synchronize()
        sampler.stop()
    return {"peak_device_mb": sampler.device_mb, "peak_host_mb": sampler.host_mb}


def neighbor_agreement(reference, indices) -> float:
    """Fraction of neighbors shared with the reference, averaged over queries."""
    reference = reference.cpu()
    indices = indices.cpu()
    hits = (reference.unsqueeze(2) == indices.unsqueeze(1)).any(dim=2).sum()
    return float(hits) / reference.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500_000, help="database size")
    parser.add_argument("--d", type=int, nargs="+", default=[64, 128], help="dimension")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1024, 8192, 65536]
    )
    parser.add_argument("--k", type=int, default=15, help="neighbors per query")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per case")
    parser.add_argument(
        "--stream-rows",
        type=int,
        nargs="*",
        default=[],
        help="extra runs with an explicit FAISS call size, where supported",
    )
    parser.add_argument(
        "--no-regroup-arm",
        action="store_true",
        help="add a run whose FAISS call size is the loader's batch size",
    )
    parser.add_argument("--index-type", type=str, nargs="+", default=["Flat"])
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", type=str, default="", help="provenance label")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")

    device = torch.cuda.current_device()
    supports_stream_rows = (
        "stream_batch_size" in inspect.signature(FaissConfig).parameters
    )

    report = {
        "n": args.n,
        "k": args.k,
        "index_types": args.index_type,
        "pin_memory": args.pin_memory,
        "repeats": args.repeats,
        "seed": args.seed,
        "label": args.label,
        "gpu": torch.cuda.get_device_name(device),
        "torchdr_path": torchdr.__file__,
        "supports_stream_batch_size": supports_stream_rows,
        "rows": [],
    }

    index_types = " ".join(args.index_type)
    print(f"n={args.n} k={args.k} index={index_types} repeats={args.repeats}")
    print(f"label={args.label or '-'} gpu={report['gpu']}")
    print(f"torchdr={report['torchdr_path']}")
    print(f"stream_batch_size supported: {supports_stream_rows}")
    print(
        ROW.format(
            "case",
            "batch",
            "stream",
            "build",
            "search",
            "total",
            "gpuMB",
            "rssMB",
            "agree",
        )
    )

    def emit(row):
        report["rows"].append(row)
        print(
            ROW.format(
                row["case"],
                row["batch"],
                row["stream"],
                f"{row['build_s']:.3f}",
                f"{row['search_s']:.3f}",
                f"{row['total_s']:.3f}",
                f"{row['peak_device_mb']:.0f}",
                f"{row['peak_host_mb']:.0f}",
                f"{row['agreement']:.3f}",
            )
        )
        if args.output:
            with open(args.output, "w") as handle:
                json.dump(report, handle, indent=2)

    for index_type in args.index_type:
        for d in args.d:
            torch.manual_seed(args.seed)
            X = torch.randn(args.n, d)

            # Reference neighbors from the in-memory tensor path, same index type,
            # so agreement measures the streaming path rather than the index.
            _, reference = pairwise_distances(
                X.cuda(),
                k=args.k,
                backend=FaissConfig(index_type=index_type),
                return_indices=True,
                exclude_diag=True,
            )
            reference = reference.cpu()
            torch.cuda.empty_cache()

            for batch in args.batch_sizes:
                loader = DataLoader(
                    TensorDataset(X),
                    batch_size=batch,
                    shuffle=False,
                    pin_memory=args.pin_memory,
                )

                stream_arms = [None]
                if supports_stream_rows:
                    if args.no_regroup_arm:
                        stream_arms.append(batch)
                    stream_arms.extend(args.stream_rows)

                for stream_rows in stream_arms:
                    kwargs = {"index_type": index_type}
                    if stream_rows is not None:
                        kwargs["stream_batch_size"] = stream_rows

                    def call(kwargs=kwargs):
                        return pairwise_distances(
                            loader,
                            k=args.k,
                            backend=FaissConfig(**kwargs),
                            return_indices=True,
                            exclude_diag=True,
                        )

                    timing, result = timed_call(call, args.repeats)
                    agreement = neighbor_agreement(reference, result[1])
                    result = None
                    torch.cuda.empty_cache()

                    memory = sampled_call(call, device)
                    torch.cuda.empty_cache()

                    emit(
                        {
                            "case": f"{index_type}/d{d}",
                            "index_type": index_type,
                            "batch": batch,
                            "stream": "auto" if stream_rows is None else stream_rows,
                            "agreement": agreement,
                            **timing,
                            **memory,
                        }
                    )

            del X, reference
            torch.cuda.empty_cache()

    if args.output:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
