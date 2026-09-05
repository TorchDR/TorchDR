"""[Perf] Peak-HBM / wall-time benchmark for ``input_layout='sharded'`` UMAP.

Compares, on a real NCCL + FAISS-GPU process group, the per-rank device memory
of a distributed UMAP fit when the raw feature rows are

  * replicated  -- every rank holds the full ``(N, d)`` input and builds an index
                   over all of it (the pre-input-sharding behavior), versus
  * sharded     -- every rank holds only its ``(N / W, d)`` shard and indexes just
                   that shard (issue #359/#308).

The embedding stays replicated in both cases, so any HBM delta comes from the
``O(N * d)`` input tensor + FAISS index footprint that input-sharding splits.
FAISS-GPU allocates outside the torch caching allocator, so device-wide peak is
sampled from ``cuda.mem_get_info`` by a background thread; the torch-allocator
peak is reported alongside as a cross-check.

Launch (not a pytest; a torchrun entrypoint)::

    torchrun --standalone --nnodes=1 --nproc-per-node=2 \\
        benchmarks/faiss/input_shard_hbm.py \\
        --samples 50000 --dim 768 --neighbors 15 --iters 100
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import argparse
import json
import os
import threading
import time

import torch
import torch.distributed as dist

from torchdr import UMAP
from torchdr.distance import FaissPlanConfig
from torchdr.distributed import init_distributed, shutdown_distributed


class _HBMSampler:
    """Poll device-wide used HBM in a background thread; track the peak."""

    def __init__(self, device, interval=0.004):
        self.device = device
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.baseline = 0
        self.total = 0
        self.peak_used = 0

    def _loop(self):
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info(self.device)
            used = total - free
            if used > self.peak_used:
                self.peak_used = used
            time.sleep(self.interval)

    def __enter__(self):
        free, total = torch.cuda.mem_get_info(self.device)
        self.total = total
        self.baseline = total - free
        self.peak_used = self.baseline
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        # One final direct read in case the peak landed between polls.
        free, total = torch.cuda.mem_get_info(self.device)
        self.peak_used = max(self.peak_used, total - free)


def _global_data(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float32)


def _shard_bounds(n, world_size, rank):
    base, rem = divmod(n, world_size)
    counts = [base + (1 if r < rem else 0) for r in range(world_size)]
    offset = sum(counts[:rank])
    return offset, counts[rank]


def _run_case(layout, X_global, rank, world_size, device, args):
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()

    if layout == "sharded":
        offset, count = _shard_bounds(args.n, world_size, rank)
        X_in = X_global[offset : offset + count].to(device)
        model = UMAP(
            n_neighbors=args.k,
            n_components=2,
            max_iter=args.iters,
            random_state=0,
            distributed=True,
            input_layout="sharded",
            backend="faiss",
            init="random",
        )
    else:
        X_in = X_global.to(device)
        model = UMAP(
            n_neighbors=args.k,
            n_components=2,
            max_iter=args.iters,
            random_state=0,
            distributed=True,
            backend=FaissPlanConfig(),
            init="random",
        )

    input_bytes = X_in.element_size() * X_in.nelement()
    torch.cuda.synchronize(device)
    dist.barrier()

    with _HBMSampler(device) as sampler:
        t0 = time.perf_counter()
        embedding = model.fit_transform(X_in)
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0

    result = {
        "layout": layout,
        "rank": rank,
        "rows_in": int(X_in.shape[0]),
        "emb_rows": int(embedding.shape[0]),
        "input_mb": input_bytes / 1024**2,
        "peak_used_mb": sampler.peak_used / 1024**2,
        "baseline_mb": sampler.baseline / 1024**2,
        "peak_over_baseline_mb": (sampler.peak_used - sampler.baseline) / 1024**2,
        "peak_torch_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_s": wall,
    }
    del model, X_in, embedding
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def _gather(result):
    bucket = [None] * dist.get_world_size()
    dist.all_gather_object(bucket, result)
    return bucket


def main():
    # NOTE: torchrun prefix-matches its own long options, so avoid arg names that
    # abbreviate to torchrun flags (e.g. ``--n`` collides with ``--nnodes``).
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000, dest="n")
    parser.add_argument("--dim", type=int, default=768, dest="d")
    parser.add_argument("--neighbors", type=int, default=15, dest="k")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    init_distributed(backend="nccl")
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise SystemExit("launch under torchrun with >=2 processes")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Warm up CUDA + FAISS so the first case does not eat context-init memory.
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)

    X_global = _global_data(args.n, args.d)
    replicated = _gather(
        _run_case("replicated", X_global, rank, world_size, device, args)
    )
    sharded = _gather(_run_case("sharded", X_global, rank, world_size, device, args))

    if rank == 0:

        def peak(rows):
            return max(r["peak_used_mb"] for r in rows)

        def wall(rows):
            return max(r["wall_s"] for r in rows)

        rep_peak, sh_peak = peak(replicated), peak(sharded)
        rep_in = max(r["input_mb"] for r in replicated)
        sh_in = max(r["input_mb"] for r in sharded)
        header = (
            f"\n=== input-shard HBM bench  N={args.n} d={args.d} "
            f"k={args.k} iters={args.iters} world_size={world_size} ==="
        )
        print(header)
        print(
            f"{'layout':<11}{'per-rank input MB':>20}{'peak HBM MB':>16}"
            f"{'peak-baseline MB':>20}{'wall s':>10}"
        )
        for name, rows in (("replicated", replicated), ("sharded", sharded)):
            print(
                f"{name:<11}{max(r['input_mb'] for r in rows):>20.1f}"
                f"{peak(rows):>16.1f}"
                f"{max(r['peak_over_baseline_mb'] for r in rows):>20.1f}"
                f"{wall(rows):>10.2f}"
            )
        print(
            f"\ninput tensor shrink : {rep_in:.1f} MB -> {sh_in:.1f} MB "
            f"({rep_in / max(sh_in, 1e-9):.2f}x)"
        )
        print(
            f"peak HBM shrink     : {rep_peak:.1f} MB -> {sh_peak:.1f} MB "
            f"({rep_peak / max(sh_peak, 1e-9):.2f}x, "
            f"{rep_peak - sh_peak:.1f} MB saved/rank)"
        )
        print("VERDICT:", "APPROVE" if sh_peak < rep_peak else "REFUTE")
        print("JSON " + json.dumps({"replicated": replicated, "sharded": sharded}))

    dist.barrier()
    shutdown_distributed()


if __name__ == "__main__":
    main()
