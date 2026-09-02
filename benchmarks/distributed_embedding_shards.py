"""Benchmark the distributed closed-form gradient exchange in neighbor embedding.

With closed-form gradients (``UMAP``), every rank owns a contiguous chunk of
rows and ``_compute_gradients()`` returns a ``(chunk, n_components)`` block for
that chunk alone.  ``AffinityMatcher._training_step`` then turns it into a
gradient for the *replicated* embedding:

    gradients (chunk, q)
      -> torch.zeros_like(embedding)          # (n, q) allocation + memset
      -> chunk_start = chunk_indices_[0].item()   # host sync, every iteration
      -> full[chunk_start : ...] = gradients  # scatter
      -> dist.all_reduce(full)                # dense (n, q) all-reduce
      -> optimizer.step()                     # SGD over the full (n, q) tensor

Because a row is only ever written by its owner, the all-reduce is a sum of one
non-zero block and ``P - 1`` zero blocks.  The obvious alternative is to keep
the optimized parameter local -- one ``(chunk, q)`` tensor per rank -- step it,
and ``all_gather_into_tensor`` the updated rows back into the replicated
embedding.  A ring all-gather moves ``(P-1)/P * n * q`` bytes against the ring
all-reduce's ``2(P-1)/P * n * q``, and the zero-fill disappears.

This script measures whether that is worth doing.  Three arms:

``all_reduce``
    The shipped path, instrumented.

``hoisted``
    Control arm.  The shipped path with ``chunk_indices_[0]`` read once instead
    of every iteration.  Two lines, no change of algorithm, and it isolates how
    much of any speed-up comes from dropping the per-iteration host sync rather
    than from changing the collective.  Bitwise identical to ``all_reduce``.

``owner_shard``
    Each rank optimizes a local ``(chunk, q)`` parameter and publishes it with
    ``all_gather_into_tensor``; unequal chunks gather fixed-width padded shards
    and compact them.  Also bitwise identical, because a zero-padded all-reduce
    reproduces the owner's gradient exactly and SGD without momentum is a
    per-row elementwise update.

Reported per rank: CUDA-event time for the negative-sampling, gradient,
assemble, collective and optimizer regions of the training step; end-to-end
``fit_transform`` wall time; peak allocated memory; optimizer-state and
parameter bytes.  ``--micro-sizes`` additionally times the two collectives on
their own, at the exact shapes the embedding uses, to separate the cost of the
collective from the time ranks spend waiting inside it.

Reading the per-rank table matters: the ``comm`` column is where load imbalance
between chunks surfaces, since the collective is a barrier.  A rank whose
``grad`` is below the maximum pays the difference in ``comm``.  Compare the
``comm`` column against the ``--micro-sizes`` numbers to tell the two apart.

Example
-------
torchrun --nproc_per_node=4 benchmarks/distributed_embedding_shards.py \
    --dataset zheng --n-samples 1306127 --max-iter 500 --reps 3 \
    --micro-sizes 1306127
"""

import argparse
import gzip
import json
import os
import pickle
import statistics
import time
from io import BytesIO

import numpy as np
import torch
import torch.distributed as dist

from torchdr import UMAP

DATASETS = {
    "macosko": "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz",
    "zheng": "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz",
}

REGIONS = ("sample", "grad", "assemble", "comm", "opt")

ARMS = ("all_reduce", "hoisted", "owner_shard")


# --- Timing ----------------------------------------------------------------


class Timer:
    """CUDA-event timer that defers synchronization to the end of the run."""

    def __init__(self):
        self.events = {name: [] for name in REGIONS}

    def region(self, name):
        return _Region(self, name)

    def summary(self, skip):
        torch.cuda.synchronize()
        out = {}
        for name, pairs in self.events.items():
            ms = [s.elapsed_time(e) for s, e in pairs[skip:]]
            out[name] = (
                None
                if not ms
                else {
                    "n": len(ms),
                    "median_ms": statistics.median(ms),
                    "mean_ms": statistics.fmean(ms),
                    "min_ms": min(ms),
                    "max_ms": max(ms),
                }
            )
        return out


class _Region:
    __slots__ = ("timer", "name", "start")

    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.timer.events[self.name].append((self.start, end))
        return False


# --- Arms ------------------------------------------------------------------


def make_arm(arm):
    """Build an instrumented ``UMAP`` subclass for one arm."""

    class Arm(UMAP):
        _arm = arm

        def _fit_transform(self, X, y=None):
            self._timer = Timer()
            self._chunk_start = None
            self._shard = None
            self._check_buffer = None
            self._footprint = None
            return super()._fit_transform(X, y)

        # -- owner_shard: optimize the locally owned rows --------------------

        def _set_params(self):
            if arm != "owner_shard" or getattr(self, "world_size", 1) == 1:
                return super()._set_params()

            n_samples, q = self.embedding_.shape
            base, remainder = divmod(n_samples, self.world_size)
            self._sizes = [
                base + (1 if r < remainder else 0) for r in range(self.world_size)
            ]
            start = int(self.chunk_indices_[0].item())
            size = int(self.chunk_indices_.numel())
            if size != self._sizes[self.rank] or start != sum(self._sizes[: self.rank]):
                raise RuntimeError(
                    "owner_shard assumes the standard contiguous row partition"
                )
            self._shard = (
                self.embedding_.data[start : start + size].clone().requires_grad_(True)
            )
            self.embedding_.requires_grad_(False)
            self._even = remainder == 0
            if not self._even:
                self._max_shard = max(self._sizes)
                self._send = torch.empty(
                    (self._max_shard, q),
                    device=self.embedding_.device,
                    dtype=self.embedding_.dtype,
                )
                self._gather = torch.empty(
                    (self._max_shard * self.world_size, q),
                    device=self.embedding_.device,
                    dtype=self.embedding_.dtype,
                )
            self.params_ = [{"params": self._shard}]
            return self.params_

        def _publish(self):
            if self._even:
                dist.all_gather_into_tensor(self.embedding_.data, self._shard.data)
                return
            self._send[: self._shard.shape[0]].copy_(self._shard.data)
            dist.all_gather_into_tensor(self._gather, self._send)
            offset = 0
            for rank, size in enumerate(self._sizes):
                padded = rank * self._max_shard
                self.embedding_.data[offset : offset + size].copy_(
                    self._gather[padded : padded + size]
                )
                offset += size

        def _expose_grad_for_check(self):
            """Give the convergence check a full-size gradient to take a norm of.

            ``_fit_transform`` reads ``embedding_.grad.norm(2)`` every
            ``check_interval`` iterations.  A real implementation would add a
            hook for the sharded norm; here the shard is scattered into a
            persistent buffer and reduced, which costs the same as one baseline
            iteration amortized over ``check_interval``.
            """
            if self._check_buffer is None:
                self._check_buffer = torch.zeros_like(self.embedding_)
            else:
                self._check_buffer.zero_()
            start = sum(self._sizes[: self.rank])
            self._check_buffer[start : start + self._shard.shape[0]] = self._shard.grad
            dist.all_reduce(self._check_buffer, op=dist.ReduceOp.SUM)
            self.embedding_.grad = self._check_buffer

        # -- instrumented step -----------------------------------------------

        def on_training_step_start(self):
            with self._timer.region("sample"):
                super().on_training_step_start()

        def _training_step(self):
            t = self._timer
            self.optimizer_.zero_grad(set_to_none=True)

            with t.region("grad"):
                gradients = self._compute_gradients()

            world = getattr(self, "world_size", 1)
            sharded = arm == "owner_shard" and world > 1

            if world == 1:
                self.embedding_.grad = gradients
            elif sharded:
                self._shard.grad = gradients
            else:
                with t.region("assemble"):
                    expected = len(self.chunk_indices_)
                    full_gradients = torch.zeros_like(self.embedding_)
                    if arm == "hoisted":
                        if self._chunk_start is None:
                            self._chunk_start = int(self.chunk_indices_[0].item())
                        chunk_start = self._chunk_start
                    else:
                        chunk_start = self.chunk_indices_[0].item()
                    full_gradients[chunk_start : chunk_start + expected] = gradients
                with t.region("comm"):
                    dist.all_reduce(full_gradients, op=dist.ReduceOp.SUM)
                self.embedding_.grad = full_gradients

            with t.region("opt"):
                self.optimizer_.step()
                if self.scheduler_ is not None:
                    self.scheduler_.step()

            if sharded:
                with t.region("comm"):
                    self._publish()
                # The convergence check reads embedding_.grad in this same
                # iteration, after _training_step returns.
                if int(self.n_iter_) % self.check_interval == 0:
                    self._expose_grad_for_check()
            return None

        def on_training_step_end(self):
            super().on_training_step_end()
            # clear_memory() drops the optimizer at the end of the fit, so the
            # footprint has to be read while the loop is still running.
            if self._footprint is None:
                self._footprint = self._measure_footprint()

        def _measure_footprint(self):
            state_bytes = sum(
                v.numel() * v.element_size()
                for state in self.optimizer_.state.values()
                for v in state.values()
                if torch.is_tensor(v)
            )
            param_bytes, n_params = 0, 0
            for group in self.optimizer_.param_groups:
                for p in group["params"]:
                    param_bytes += p.numel() * p.element_size()
                    n_params += p.numel()
                    if p.grad is not None:
                        param_bytes += p.grad.numel() * p.grad.element_size()
            emb = self.embedding_
            return {
                "optimizer_state_bytes": state_bytes,
                "param_plus_grad_bytes": param_bytes,
                "replicated_embedding_bytes": emb.numel() * emb.element_size(),
                "n_optimized_params": n_params,
            }

    Arm.__name__ = f"UMAP_{arm}"
    return Arm


# --- Data ------------------------------------------------------------------


def load_dataset(name, cache_dir, n_samples, seed):
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, f"{name}_pca50.npy")
    if os.path.exists(cached):
        X = np.load(cached)
    else:
        import requests

        response = requests.get(DATASETS[name], stream=True)
        response.raise_for_status()
        with gzip.open(BytesIO(response.content), "rb") as handle:
            X = pickle.load(handle)["pca_50"].astype("float32")
        np.save(cached, X)
    if 0 < n_samples < X.shape[0]:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(X.shape[0], size=n_samples, replace=False))
        X = np.ascontiguousarray(X[idx])
    return X


# --- Collective microbenchmark ---------------------------------------------


def collective_microbench(n, q, dtype, reps, warmup, device):
    """Time the two collectives on their own, at the embedding's shapes."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    base, rem = divmod(n, world)
    sizes = [base + (1 if r < rem else 0) for r in range(world)]
    my_size, max_chunk = sizes[rank], max(sizes)

    full = torch.zeros((n, q), device=device, dtype=dtype)
    shard = torch.zeros((my_size, q), device=device, dtype=dtype)
    even = rem == 0
    gather_out = (
        full
        if even
        else torch.empty((max_chunk * world, q), device=device, dtype=dtype)
    )
    gather_in = (
        shard if even else torch.empty((max_chunk, q), device=device, dtype=dtype)
    )

    def timed(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        for _ in range(reps):
            dist.barrier()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        return {
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
        }

    def full_path():
        buf = torch.zeros((n, q), device=device, dtype=dtype)
        buf[0:my_size] = shard
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    elem = torch.tensor([], dtype=dtype).element_size()
    return {
        "n": n,
        "q": q,
        "world_size": world,
        "even_split": even,
        "payload_bytes": n * q * elem,
        "ring_allreduce_moved_bytes": int(2 * (world - 1) / world * n * q * elem),
        "ring_allgather_moved_bytes": int((world - 1) / world * n * q * elem),
        "all_reduce_ms": timed(lambda: dist.all_reduce(full, op=dist.ReduceOp.SUM)),
        "zeros_scatter_all_reduce_ms": timed(full_path),
        "all_gather_into_tensor_ms": timed(
            lambda: dist.all_gather_into_tensor(gather_out, gather_in)
        ),
    }


# --- Driver ----------------------------------------------------------------


def run_arm(arm, args, X, device, rank, world):
    cls = make_arm(arm)

    def build():
        return cls(
            n_neighbors=args.n_neighbors,
            max_iter=args.max_iter,
            device=device,
            backend="faiss",
            random_state=args.seed,
            verbose=False,
        )

    for _ in range(args.warmup):
        torch.cuda.empty_cache()
        build().fit_transform(X)
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()

    walls, timings, mem, footprint = [], [], [], None
    embedding = None
    for rep in range(args.reps):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build()
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.fit_transform(X)
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        walls.append(time.perf_counter() - start)
        mem.append(torch.cuda.max_memory_allocated() / 2**20)
        timings.append(model._timer.summary(args.skip_iters))
        footprint = footprint or model._footprint
        if rep == 0:
            embedding = torch.as_tensor(out).detach().float().cpu().clone()
        del model, out
        if rank == 0:
            print(f"  [{arm}] rep {rep}: {walls[-1]:.4f} s", flush=True)

    return {
        "arm": arm,
        "wall_s": walls,
        "wall_median_s": statistics.median(walls),
        "peak_allocated_mb": statistics.median(mem),
        "timings": timings,
        "footprint": footprint,
    }, embedding


def region_medians(result):
    """Median across reps of each rep's per-iteration median, per region."""
    out = {}
    for region in REGIONS:
        vals = [t[region]["median_ms"] for t in result["timings"] if t.get(region)]
        out[region] = statistics.median(vals) if vals else 0.0
    return out


def report(record):
    rows = []
    for arm, res in record["arms"].items():
        med = region_medians(res)
        med["step"] = sum(med.values())
        rows.append((arm, med, res))

    print("\n### per-rank training step (ms/iteration, median)\n")
    header = "| arm | " + " | ".join(REGIONS) + " | step | wall s | peak MB |"
    print(header)
    print("|" + "---|" * (len(REGIONS) + 4))
    for arm, med, res in rows:
        cells = " | ".join(f"{med[r]:.4f}" for r in REGIONS)
        print(
            f"| {arm} | {cells} | {med['step']:.4f} | "
            f"{res['wall_median_s']:.3f} | {res['peak_allocated_mb']:.1f} |"
        )

    base = rows[0][2]["wall_median_s"]
    print("\n### end-to-end vs the shipped path\n")
    if not record["config"]["warmup"]:
        print(
            "> Run with `--warmup 0`: the first arm absorbs CUDA context and "
            "FAISS setup, so the wall-time deltas below are not meaningful. "
            "Use the per-iteration step column, or re-run with `--warmup 1`.\n"
        )
    print("| arm | wall s | delta | optimizer state B | param+grad MiB |")
    print("|---|---|---|---|---|")
    for arm, _, res in rows:
        fp = res["footprint"] or {}
        delta = 100.0 * (res["wall_median_s"] - base) / base
        print(
            f"| {arm} | {res['wall_median_s']:.3f} | {delta:+.2f}% | "
            f"{fp.get('optimizer_state_bytes', 0)} | "
            f"{fp.get('param_plus_grad_bytes', 0) / 2**20:.3f} |"
        )

    if record["equivalence"]:
        print("\n### embedding equivalence vs the shipped path\n")
        for arm, eq in record["equivalence"].items():
            print(
                f"- {arm}: max |delta| = {eq['max_abs_diff']:.3e}, "
                f"bitwise equal = {eq['bitwise_equal']}"
            )

    if record["micro"]:
        print("\n### collectives in isolation (ms, median)\n")
        print(
            "| n | world | even | payload MiB | all_reduce | "
            "zeros+scatter+all_reduce | all_gather | recoverable |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for m in record["micro"]:
            fullpath = m["zeros_scatter_all_reduce_ms"]["median"]
            gather = m["all_gather_into_tensor_ms"]["median"]
            print(
                f"| {m['n']} | {m['world_size']} | {m['even_split']} | "
                f"{m['payload_bytes'] / 2**20:.2f} | "
                f"{m['all_reduce_ms']['median']:.4f} | {fullpath:.4f} | "
                f"{gather:.4f} | {fullpath - gather:.4f} |"
            )
        step = rows[0][1]["step"]
        if step:
            best = max(
                m["zeros_scatter_all_reduce_ms"]["median"]
                - m["all_gather_into_tensor_ms"]["median"]
                for m in record["micro"]
            )
            print(
                f"\nUpper bound on what replacing the collective can save: "
                f"{best:.4f} ms of a {step:.4f} ms step ({100 * best / step:.2f}%)."
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), default="zheng")
    p.add_argument("--n-samples", type=int, default=200000)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--skip-iters",
        type=int,
        default=10,
        help="leading iterations dropped from the region medians",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="benchmarks/data")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument(
        "--micro-sizes",
        default="",
        help="comma-separated n values for the isolated collective timings",
    )
    p.add_argument("--json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    record = {
        "world_size": world,
        "rank": rank,
        "device_name": torch.cuda.get_device_name(local_rank),
        "torch": torch.__version__,
        "config": vars(args),
        "micro": [],
        "arms": {},
        "equivalence": {},
    }

    for n in [int(v) for v in args.micro_sizes.split(",") if v]:
        record["micro"].append(
            collective_microbench(n, 2, torch.float32, 20, 5, device)
        )

    X = load_dataset(args.dataset, args.cache_dir, args.n_samples, args.seed)
    record["n_samples"] = int(X.shape[0])
    if rank == 0:
        print(f"{args.dataset}: {X.shape}, world_size={world}", flush=True)

    embeddings = {}
    for arm in args.arms.split(","):
        res, emb = run_arm(arm, args, X, device, rank, world)
        record["arms"][arm] = res
        embeddings[arm] = emb

    names = list(embeddings)
    for other in names[1:]:
        diff = (embeddings[other] - embeddings[names[0]]).abs()
        record["equivalence"][other] = {
            "vs": names[0],
            "max_abs_diff": float(diff.max()),
            "bitwise_equal": bool(torch.equal(embeddings[other], embeddings[names[0]])),
        }

    if rank == 0:
        report(record)
    if args.json:
        out = args.json.replace(".json", f".rank{rank}.json")
        with open(out, "w") as handle:
            json.dump(record, handle, indent=2)
        if rank == 0:
            print(f"\nwrote {out}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
