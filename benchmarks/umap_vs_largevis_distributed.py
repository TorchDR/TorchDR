"""Test UMAP (direct gradients) and LargeVis (autograd) in distributed mode.

Compare neighborhood preservation scores for both methods.
"""

import os
import time
import gzip
import pickle
from io import BytesIO

import requests
import torch
import torch.distributed as dist

from torchdr import UMAP, LargeVis
from torchdr.eval import neighborhood_preservation


def setup_distributed():
    """Initialize distributed training if launched with torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_zheng_dataset():
    """Load Zheng et al. 2017 10x mouse dataset (1.3M cells, 50 PCA dims)."""
    url = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with gzip.open(BytesIO(response.content), "rb") as f:
        data = pickle.load(f)

    X = torch.from_numpy(data["pca_50"].astype("float32"))
    return X


def main():
    is_distributed, rank, world_size = setup_distributed()

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("UMAP vs LargeVis Distributed Comparison")
        print("Dataset: Zheng et al. 2017 (10x Mouse, 1.3M cells)")
        print(f"Configuration: {world_size} GPU{'s' if world_size > 1 else ''}")
        print(f"{'=' * 70}\n")

    if rank == 0:
        print("Loading dataset...")

    X = load_zheng_dataset()
    n_samples, n_features = X.shape

    if rank == 0:
        print(f"  Samples: {n_samples:,}")
        print(f"  Features: {n_features}\n")

    if is_distributed:
        dist.barrier()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== UMAP (uses direct gradients) =====
    if rank == 0:
        print(f"{'=' * 70}")
        print("UMAP (Direct Gradients)")
        print(f"{'=' * 70}")
        print("Computing embedding...")

    start_time = time.time()
    umap = UMAP(
        n_components=2,
        n_neighbors=30,
        max_iter=1000,
        device=device,
        backend="faiss",
        verbose=True if rank == 0 else False,
    )
    Z_umap = umap.fit_transform(X)
    umap_time = time.time() - start_time

    if rank == 0:
        print(f"  Embedding time: {umap_time:.2f}s")
        print("  Computing neighborhood preservation...")

    start_time = time.time()
    np_score_umap = neighborhood_preservation(
        X,
        Z_umap,
        K=100,
        metric="euclidean",
        backend="faiss",
        device=device,
        distributed="auto",
        return_per_sample=False,
    )
    np_time_umap = time.time() - start_time

    if is_distributed:
        dist.barrier()

    # ===== LargeVis (uses autograd) =====
    if rank == 0:
        print(f"\n{'=' * 70}")
        print("LargeVis (Autograd)")
        print(f"{'=' * 70}")
        print("Computing embedding...")

    start_time = time.time()
    largevis = LargeVis(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        device=device,
        backend="faiss",
        verbose=True if rank == 0 else False,
    )
    Z_largevis = largevis.fit_transform(X)
    largevis_time = time.time() - start_time

    if rank == 0:
        print(f"  Embedding time: {largevis_time:.2f}s")
        print("  Computing neighborhood preservation...")

    start_time = time.time()
    np_score_largevis = neighborhood_preservation(
        X,
        Z_largevis,
        K=100,
        metric="euclidean",
        backend="faiss",
        device=device,
        distributed="auto",
        return_per_sample=False,
    )
    np_time_largevis = time.time() - start_time

    # ===== Results =====
    if is_distributed:
        # Gather results
        umap_scores = [None] * world_size
        largevis_scores = [None] * world_size
        umap_np_times = [None] * world_size
        largevis_np_times = [None] * world_size

        umap_cpu = (
            np_score_umap.cpu().item()
            if torch.is_tensor(np_score_umap)
            else np_score_umap
        )
        largevis_cpu = (
            np_score_largevis.cpu().item()
            if torch.is_tensor(np_score_largevis)
            else np_score_largevis
        )

        dist.gather_object(umap_cpu, umap_scores if rank == 0 else None, dst=0)
        dist.gather_object(largevis_cpu, largevis_scores if rank == 0 else None, dst=0)
        dist.gather_object(np_time_umap, umap_np_times if rank == 0 else None, dst=0)
        dist.gather_object(
            np_time_largevis, largevis_np_times if rank == 0 else None, dst=0
        )

        if rank == 0:
            avg_umap = sum(umap_scores) / len(umap_scores)
            avg_largevis = sum(largevis_scores) / len(largevis_scores)
            max_np_time_umap = max(umap_np_times)
            max_np_time_largevis = max(largevis_np_times)

            print(f"\n{'=' * 70}")
            print("Results Summary")
            print(f"{'=' * 70}")
            print(f"\nUMAP (Direct Gradients):")
            print(f"  Embedding time: {umap_time:.2f}s")
            print(f"  Neighborhood preservation: {avg_umap:.4f}")
            print(f"  NP eval time: {max_np_time_umap:.2f}s")

            print(f"\nLargeVis (Autograd):")
            print(f"  Embedding time: {largevis_time:.2f}s")
            print(f"  Neighborhood preservation: {avg_largevis:.4f}")
            print(f"  NP eval time: {max_np_time_largevis:.2f}s")

            print(f"\nComparison:")
            print(f"  UMAP vs LargeVis NP score: {avg_umap:.4f} vs {avg_largevis:.4f}")
            print(f"  Difference: {abs(avg_umap - avg_largevis):.4f}")
            print(f"{'=' * 70}\n")

        cleanup_distributed()
    else:
        print(f"\n{'=' * 70}")
        print("Results Summary")
        print(f"{'=' * 70}")
        print(f"\nUMAP (Direct Gradients):")
        print(f"  Embedding time: {umap_time:.2f}s")
        print(f"  Neighborhood preservation: {np_score_umap:.4f}")
        print(f"  NP eval time: {np_time_umap:.2f}s")

        print(f"\nLargeVis (Autograd):")
        print(f"  Embedding time: {largevis_time:.2f}s")
        print(f"  Neighborhood preservation: {np_score_largevis:.4f}")
        print(f"  NP eval time: {np_time_largevis:.2f}s")

        print(f"\nComparison:")
        print(
            f"  UMAP vs LargeVis NP score: {np_score_umap:.4f} vs {np_score_largevis:.4f}"
        )
        print(f"  Difference: {abs(np_score_umap - np_score_largevis):.4f}")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
