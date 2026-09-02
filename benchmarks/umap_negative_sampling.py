"""Benchmark the UMAP repulsive (negative-sampling) branch.

``UMAP`` draws ``n_negatives = negative_sample_rate * n_neighbors`` negatives for
every row of the local chunk, evaluates the full ``(chunk, n_negatives)``
rectangle, and then zeroes the columns beyond each row's per-iteration budget
``min(active_positive_edges * negative_sample_rate, n_negatives)``.  Only a
minority of the rectangle therefore contributes to the gradient.

This script measures how much is actually left on the table.  It compares three
implementations of ``_compute_repulsive_gradients`` under identical conditions:

``baseline``
    The shipped kernel.  ``pairwise_distances_indexed`` builds the
    ``(chunk, n_negatives, n_components)`` difference tensor to produce the
    squared distances, and the gradient block then rebuilds the same tensor.

``fused``
    Control arm.  Still fully rectangular and still consumes the same eager
    draw, but materializes the difference tensor once and derives the distances
    from it.  Same arithmetic in the same order, so it is *bitwise* identical to
    the baseline; it isolates how much of any speed-up comes from removing the
    redundant materialization rather than from packing.

``packed``
    Evaluates only ``sum_i min(active_i * negative_sample_rate, n_negatives)``
    negative pairs in a flat layout, draws them with a single ``searchsorted``
    over a globally offset exclusion table, and scatters the result back with
    ``index_add_``.

Reported per arm: end-to-end ``fit_transform`` wall time (median and range over
``--reps`` runs), CUDA-event time for the attractive branch, the repulsive
branch and the whole training step, peak allocated bytes per phase, the negative
pairs actually evaluated, and K-ary neighborhood preservation.

``--check`` additionally replays every arm against one identical model state and
reports the gradient difference against the baseline, plus a uniformity and
exclusion audit of the packed draw.

Example
-------
python benchmarks/umap_negative_sampling.py --dataset zheng --n-samples 200000 \
    --n-neighbors 30 --max-iter 500 --reps 3 --check
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

from torchdr import UMAP
from torchdr.distance import pairwise_distances_indexed
from torchdr.eval import neighborhood_preservation
from torchdr.neighbor_embedding.base import NeighborEmbedding

DATASETS = {
    "macosko": "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz",
    "zheng": "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz",
}


# --- Repulsive-branch arms -------------------------------------------------
#
# Written as plain functions of a fitted model so that the --check probe can
# evaluate every arm against one identical state.


def negative_counts(model):
    """Per-row number of negatives the baseline actually uses this iteration."""
    active = model.mask_affinity_in_.sum(dim=1)
    return (active * model.negative_sample_rate).clamp(max=model.n_negatives).long()


def _coefficients(D, model):
    """In-place ``-2 b / ((eps + d) (1 + a d^b))`` used by all arms."""
    D_ = 1 + model._a * D**model._b
    D.add_(model._eps)
    D.mul_(D_)
    D.reciprocal_().mul_(-2 * model._b)
    return D


def repulsive_baseline(model):
    """Verbatim copy of the shipped kernel, kept so --check can audit the copy."""
    D = pairwise_distances_indexed(
        model.embedding_,
        query_indices=model.chunk_indices_,
        key_indices=model.neg_indices_,
        metric="sqeuclidean",
    )
    _coefficients(D, model)

    neg_counts = (model.mask_affinity_in_.sum(dim=1) * model.negative_sample_rate).to(
        torch.long
    )
    col_idx = torch.arange(model.n_negatives, device=model.embedding_.device)
    D.masked_fill_(col_idx[None, :].ge(neg_counts[:, None]), 0)

    diff = (
        model.embedding_[model.chunk_indices_].unsqueeze(1)
        - model.embedding_[model.neg_indices_]
    )
    grad = torch.einsum("ijk,ij->ik", diff, D)
    grad.clamp_(-4, 4)
    return grad


def repulsive_fused(model):
    """Rectangular, but the difference tensor is materialized only once."""
    diff = (
        model.embedding_[model.chunk_indices_].unsqueeze(1)
        - model.embedding_[model.neg_indices_]
    )
    D = _coefficients((diff**2).sum(dim=-1), model)

    neg_counts = (model.mask_affinity_in_.sum(dim=1) * model.negative_sample_rate).to(
        torch.long
    )
    col_idx = torch.arange(model.n_negatives, device=model.embedding_.device)
    D.masked_fill_(col_idx[None, :].ge(neg_counts[:, None]), 0)

    grad = torch.einsum("ijk,ij->ik", diff, D)
    grad.clamp_(-4, 4)
    return grad


def _flat_exclusion_table(model):
    """Flatten the per-row exclusion table into one globally sorted array.

    Row ``r`` is offset by ``r * big`` with ``big = n_candidates + 1``, so the
    per-row blocks are disjoint and ordered.  A single ``searchsorted`` over the
    flat array then returns ``r * width + shift_r``, reproducing the row-wise
    search of ``_draw_with_exclusions`` without an ``(n_pairs, width)`` temporary.
    """
    cached = getattr(model, "_flat_exclusion", None)
    if cached is not None:
        return cached
    adjusted = model.negative_adjusted_exclusion_
    rows, width = adjusted.shape
    big = int(model.n_samples_in_) + 1
    offsets = torch.arange(rows, device=adjusted.device, dtype=torch.long) * big
    flat = (adjusted + offsets.unsqueeze(1)).reshape(-1).contiguous()
    model._flat_exclusion = (flat, width, big)
    return model._flat_exclusion


def packed_draw(model, row_ids):
    """Draw one negative per entry of ``row_ids``, respecting row exclusions."""
    flat, width, big = _flat_exclusion_table(model)
    compressed = (
        torch.rand(row_ids.numel(), device=row_ids.device)
        * model.negative_available_counts_[row_ids]
    ).long()
    pos = torch.searchsorted(flat, compressed + row_ids * big, right=True)
    return compressed + (pos - row_ids * width)


def _replay_draw(model, row_ids, counts):
    """Reuse the baseline's own negatives, to isolate the packing itself."""
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(row_ids.numel(), device=row_ids.device) - starts[row_ids]
    return model.neg_indices_[row_ids, within]


def repulsive_packed(model, replay=False):
    """Evaluate only the negative pairs that survive the per-row budget."""
    device = model.embedding_.device
    counts = negative_counts(model)
    chunk = model.chunk_indices_.numel()
    grad = torch.zeros(
        (chunk, model.n_components), device=device, dtype=model.embedding_.dtype
    )

    row_ids = torch.repeat_interleave(
        torch.arange(chunk, device=device, dtype=torch.long), counts
    )
    if row_ids.numel() == 0:
        return grad
    neg_ids = (
        _replay_draw(model, row_ids, counts) if replay else packed_draw(model, row_ids)
    )

    diff = model.embedding_[model.chunk_indices_[row_ids]] - model.embedding_[neg_ids]
    D = _coefficients((diff**2).sum(dim=-1), model)

    grad.index_add_(0, row_ids, diff * D.unsqueeze(1))
    grad.clamp_(-4, 4)
    return grad


class FusedUMAP(UMAP):
    def _compute_repulsive_gradients(self):
        return repulsive_fused(self)


class PackedUMAP(UMAP):
    def on_training_step_start(self):
        # Skip the eager rectangular draw: the packed budget is only known once
        # the attractive branch has refreshed ``mask_affinity_in_``.
        NeighborEmbedding.on_training_step_start(self)

    def _compute_repulsive_gradients(self):
        return repulsive_packed(self)


ARMS = {"baseline": UMAP, "fused": FusedUMAP, "packed": PackedUMAP}


# --- Instrumentation -------------------------------------------------------


def make_timed(cls):
    """Wrap an arm with CUDA-event timing and per-phase peak-memory tracking."""

    class Timed(cls):
        def _init_embedding(self, X):
            embedding = super()._init_embedding(X)
            cap = self.max_iter + 1
            self._i = 0
            self._events = {
                key: [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(cap)
                ]
                for key in ("attractive", "repulsive", "step")
            }
            self._peak = {key: [] for key in ("attractive", "repulsive")}
            self._pairs = torch.zeros(cap, dtype=torch.long, device=self.device_)
            self._row_max = torch.zeros(cap, dtype=torch.long, device=self.device_)
            return embedding

        def _compute_attractive_gradients(self):
            # ``reset_peak_memory_stats`` sets the peak to the currently
            # allocated bytes, so the per-phase maxima below are comparable and
            # their maximum is the exact peak of the whole step.
            torch.cuda.reset_peak_memory_stats()
            self._events["attractive"][self._i][0].record()
            grad = super()._compute_attractive_gradients()
            self._events["attractive"][self._i][1].record()
            self._peak["attractive"].append(torch.cuda.max_memory_allocated())
            return grad

        def _compute_repulsive_gradients(self):
            torch.cuda.reset_peak_memory_stats()
            self._events["repulsive"][self._i][0].record()
            grad = super()._compute_repulsive_gradients()
            self._events["repulsive"][self._i][1].record()
            self._peak["repulsive"].append(torch.cuda.max_memory_allocated())
            counts = negative_counts(self)
            self._pairs[self._i] = counts.sum()
            # Unclamped, to show whether a per-step maximum-width truncation of
            # the rectangle would save anything.
            self._row_max[self._i] = (
                self.mask_affinity_in_.sum(dim=1) * self.negative_sample_rate
            ).max()
            # Captured here: clear_memory() drops chunk_indices_ before
            # fit_transform returns.
            self._rect = self.chunk_indices_.numel() * self.n_negatives
            self._n_negatives = self.n_negatives
            return grad

        def _training_step(self):
            self._events["step"][self._i][0].record()
            out = super()._training_step()
            self._events["step"][self._i][1].record()
            self._i += 1
            return out

        def report(self):
            torch.cuda.synchronize()
            n = self._i
            times = {
                key: [ev[i][0].elapsed_time(ev[i][1]) for i in range(n)]
                for key, ev in self._events.items()
            }
            peak = {k: int(np.median(v[:n])) for k, v in self._peak.items()}
            peak["step"] = max(peak["attractive"], peak["repulsive"])
            row_max = self._row_max[:n].cpu().numpy()
            return {
                "n_iter": n,
                "ms": {k: float(np.median(v)) for k, v in times.items()},
                "peak_bytes": peak,
                "neg_pairs": int(np.median(self._pairs[:n].cpu().numpy())),
                "neg_pairs_rect": int(self._rect),
                "row_max_median": int(np.median(row_max)),
                "saturated_iters": float(np.mean(row_max >= self._n_negatives)),
            }

    Timed.__name__ = f"Timed{cls.__name__}"
    return Timed


# --- Correctness probe -----------------------------------------------------


def make_probe(iters):
    """UMAP that compares every arm against the shipped kernel at ``iters``."""

    class Probe(UMAP):
        records = []

        def _compute_repulsive_gradients(self):
            grad = super()._compute_repulsive_gradients()
            step = int(self.n_iter_.item())
            if step in iters:
                self.records.append(self._probe(grad, step))
            return grad

        def _probe(self, reference, step):
            counts = negative_counts(self)
            packed = repulsive_packed(self, replay=True)
            fused = repulsive_fused(self)
            record = {
                "iter": step,
                "neg_pairs_packed": int(counts.sum().item()),
                "neg_pairs_rect": int(self.chunk_indices_.numel() * self.n_negatives),
                "copy_bitwise_equal": bool(
                    torch.equal(repulsive_baseline(self), reference)
                ),
                "fused_bitwise_equal": bool(torch.equal(fused, reference)),
                "packed_replay_max_abs_err": float(
                    (packed - reference).abs().max().item()
                ),
                "packed_replay_repeat_max_abs_diff": float(
                    (repulsive_packed(self, replay=True) - packed).abs().max().item()
                ),
            }
            record.update(self._audit_draw())
            return record

        def _audit_draw(self, rows=256, draws=512):
            """Check the packed draw never returns an excluded index."""
            device = self.embedding_.device
            chunk = self.chunk_indices_.numel()
            probe_rows = torch.arange(min(rows, chunk), device=device)
            row_ids = probe_rows.repeat_interleave(draws)
            drawn = packed_draw(self, row_ids)
            excluded = self.negative_exclusion_indices_[row_ids]
            hits = int((drawn.unsqueeze(1) == excluded).any(dim=1).sum().item())
            in_range = int(
                ((drawn >= 0) & (drawn < int(self.n_samples_in_))).sum().item()
            )
            return {
                "draw_samples": int(drawn.numel()),
                "draw_excluded_hits": hits,
                "draw_out_of_range": int(drawn.numel()) - in_range,
            }

    Probe.records = []
    return Probe


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


# --- Driver ----------------------------------------------------------------


def build(cls, args):
    return cls(
        n_neighbors=args.n_neighbors,
        max_iter=args.max_iter,
        device="cuda",
        backend="faiss",
        random_state=args.seed,
        verbose=False,
    )


def run_arm(name, args, X):
    cls = make_timed(ARMS[name])
    walls, reports, embedding = [], [], None
    for rep in range(args.reps):
        torch.cuda.empty_cache()
        model = build(cls, args)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.fit_transform(X)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - start)
        reports.append(model.report())
        if rep == 0:
            embedding = out.detach().clone()
        del model
        print(f"  [{name}] rep {rep}: {walls[-1]:.4f} s", flush=True)
    median = sorted(range(len(walls)), key=lambda i: walls[i])[len(walls) // 2]
    return {
        "wall_s": walls,
        "wall_median_s": statistics.median(walls),
        "wall_min_s": min(walls),
        "wall_max_s": max(walls),
        "timing": reports[median],
    }, embedding


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="macosko")
    parser.add_argument("--cache-dir", default="benchmark_data")
    parser.add_argument("--n-samples", type=int, default=0, help="0 keeps all rows")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quality-k", type=int, default=15)
    parser.add_argument("--arms", default="baseline,fused,packed")
    parser.add_argument("--check", action="store_true", help="run the gradient probe")
    parser.add_argument("--json", default=None, help="write full results here")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")

    X = torch.from_numpy(
        load_dataset(args.dataset, args.cache_dir, args.n_samples, args.seed)
    ).cuda()
    print(f"{args.dataset}: {tuple(X.shape)} on {torch.cuda.get_device_name(0)}")

    if args.check:
        iters = {0, 1, args.max_iter // 2, args.max_iter - 1}
        probe = build(make_probe(iters), args)
        probe.fit_transform(X)
        print("\n=== gradient probe ===")
        for record in probe.records:
            print(json.dumps(record))
        del probe
        torch.cuda.empty_cache()

    # Warm up FAISS and cuBLAS so the first timed arm is not penalized.
    warmup = build(make_timed(UMAP), args)
    warmup.max_iter = 20
    warmup.fit_transform(X)
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()

    results, quality = {}, {}
    for name in args.arms.split(","):
        print(f"[arm] {name}", flush=True)
        results[name], embedding = run_arm(name, args, X)
        quality[name] = float(
            neighborhood_preservation(
                X, embedding, K=args.quality_k, backend="faiss", device="cuda"
            )
        )
        print(f"  [{name}] preservation@{args.quality_k} = {quality[name]:.6f}")
        del embedding
        torch.cuda.empty_cache()

    base = results.get(args.arms.split(",")[0])
    header = (
        f"| arm | wall (s) | vs base | step (ms) | repulsive (ms) | "
        f"peak step (MB) | peak repulsive (MB) | preservation@{args.quality_k} |"
    )
    print("\n" + header)
    print("|" + "---|" * 8)
    for name, result in results.items():
        timing = result["timing"]
        delta = 100 * (result["wall_median_s"] / base["wall_median_s"] - 1)
        print(
            f"| {name} "
            f"| {result['wall_median_s']:.3f} "
            f"[{result['wall_min_s']:.3f}, {result['wall_max_s']:.3f}] "
            f"| {delta:+.2f}% "
            f"| {timing['ms']['step']:.3f} "
            f"| {timing['ms']['repulsive']:.3f} "
            f"| {timing['peak_bytes']['step'] / 1e6:.0f} "
            f"| {timing['peak_bytes']['repulsive'] / 1e6:.0f} "
            f"| {quality[name]:.6f} |"
        )
    timing = base["timing"]
    rect, packed = timing["neg_pairs_rect"], timing["neg_pairs"]
    print(
        f"\nnegative pairs per iteration (median): packed {packed:,} of "
        f"rectangular {rect:,} ({100 * packed / rect:.1f}%)"
    )
    # A cheaper alternative to packing is to truncate the rectangle to the
    # widest per-row budget of the step.  It only helps when that width is
    # below n_negatives, which hub rows in the symmetrized graph rarely allow.
    print(
        f"widest per-row budget (median): {timing['row_max_median']:,}; "
        f"saturates n_negatives in {100 * timing['saturated_iters']:.2f}% of "
        f"iterations"
    )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "config": vars(args),
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "n_samples": int(X.shape[0]),
                    "results": results,
                    "quality": quality,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
