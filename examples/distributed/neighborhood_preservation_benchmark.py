"""Multi-GPU neighborhood preservation benchmark.

This example demonstrates how to use TorchDR's neighborhood preservation metric
in distributed mode across multiple GPUs. The metric measures how well local
neighborhood structure is preserved when reducing dimensionality.

Dataset: Zheng et al. 2017 - 10x Mouse (1.3M single cells)
Metric: K-ary neighborhood preservation (Jaccard similarity of k-NN sets)

Usage:
    # Single GPU
    python neighborhood_preservation_benchmark.py

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 neighborhood_preservation_benchmark.py

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 neighborhood_preservation_benchmark.py
"""

import os
import time
import gzip
import pickle
from io import BytesIO

import requests
import torch
import torch.distributed as dist

from torchdr import UMAP
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
        print("Neighborhood Preservation Benchmark")
        print("Dataset: Zheng et al. 2017 (10x Mouse, 1.3M cells)")
        print(f"Configuration: {world_size} GPU{'s' if world_size > 1 else ''}")
        print(f"{'=' * 70}\n")

    if rank == 0:
        print("Loading dataset and computing embedding...")

    X = load_zheng_dataset()
    n_samples, n_features = X.shape

    umap = UMAP(
        n_components=2,
        n_neighbors=30,
        max_iter=500,
        device="cuda",
        backend="faiss",
        verbose=False,
    )
    Z = umap.fit_transform(X)

    if rank == 0:
        print(f"  Original: {n_samples:,} samples, {n_features} features")
        print(f"  Embedding: {Z.shape[0]:,} samples, {Z.shape[1]} features\n")

    if is_distributed:
        dist.barrier()

    K = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"Computing neighborhood preservation (K={K})...")
        print("  Backend: FAISS")
        print(f"  Device: {device}")
        print(f"  Distributed: {'Yes' if is_distributed else 'No'}\n")

    start_time = time.time()

    score = neighborhood_preservation(
        X,
        Z,
        K=K,
        metric="euclidean",
        backend="faiss",
        device=device,
        distributed="auto",
        return_per_sample=False,
    )

    elapsed_time = time.time() - start_time

    if is_distributed:
        all_scores = [None] * world_size
        all_times = [None] * world_size

        score_cpu = score.cpu().item() if torch.is_tensor(score) else score

        dist.gather_object(score_cpu, all_scores if rank == 0 else None, dst=0)
        dist.gather_object(elapsed_time, all_times if rank == 0 else None, dst=0)

        if rank == 0:
            global_score = sum(all_scores) / len(all_scores)
            max_time = max(all_times)

            print(f"{'=' * 70}")
            print("Results:")
            print(f"  Neighborhood preservation: {global_score:.4f}")
            print(f"  Total time: {max_time:.2f}s")
            print("  Per-GPU times:")
            for gpu_rank, t in enumerate(all_times):
                print(f"    GPU {gpu_rank}: {t:.2f}s")
            print(f"{'=' * 70}\n")

        cleanup_distributed()
    else:
        print(f"{'=' * 70}")
        print("Results:")
        print(f"  Neighborhood preservation: {score:.4f}")
        print(f"  Total time: {elapsed_time:.2f}s")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
