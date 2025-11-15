"""Multi-GPU neighborhood preservation benchmark.

This example demonstrates how to use TorchDR's neighborhood preservation metric
in distributed mode across multiple GPUs. The metric measures how well
k-nearest neighbor structure is preserved between high and low dimensional spaces.

Dataset: Zheng et al. 2017 - 10x Mouse (1.3M single cells)
Metric: Neighborhood preservation (Jaccard similarity of k-NN sets)

Usage:
    # Single GPU
    python neighborhood_preservation_benchmark.py

    # Multi-GPU (2 GPUs)
    torchdr run --gpus 2 neighborhood_preservation_benchmark.py

    # Multi-GPU (all available GPUs)
    torchdr run neighborhood_preservation_benchmark.py
"""

import time
import gzip
import pickle
from io import BytesIO

import requests
import torch

from torchdr import UMAP
from torchdr.eval import neighborhood_preservation
from torchdr.distributed import is_distributed, get_rank, get_world_size


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
    rank = get_rank()
    world_size = get_world_size()

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("Neighborhood Preservation Benchmark")
        print("Dataset: Zheng et al. 2017 (10x Mouse, 1.3M cells)")
        print(f"Configuration: {world_size} GPU{'s' if world_size > 1 else ''}")
        print(f"{'=' * 70}\n")

    if rank == 0:
        print("Loading dataset...")

    X = load_zheng_dataset()
    n_samples, n_features = X.shape

    if rank == 0:
        print(f"  Original: {n_samples:,} samples, {n_features} features\n")

    # Compute UMAP embedding
    if rank == 0:
        print("Computing UMAP embedding...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    umap = UMAP(
        n_components=2,
        n_neighbors=30,
        max_iter=500,
        device=device,
        backend="faiss",
        verbose=True if rank == 0 else False,
    )

    start_embed = time.time()
    Z = umap.fit_transform(X)
    embed_time = time.time() - start_embed

    if rank == 0:
        print(f"  Embedding time: {embed_time:.2f}s")
        print(f"  Embedding: {Z.shape[0]:,} samples, {Z.shape[1]} features\n")

    # Compute neighborhood preservation
    K = 100

    if rank == 0:
        print(f"Computing neighborhood preservation (K={K})...")
        print("  Backend: FAISS")
        print(f"  Device: {device}")
        print(f"  Distributed: {'Yes' if is_distributed() else 'No'}\n")

    start_eval = time.time()

    np_score = neighborhood_preservation(
        X,
        Z,
        K=K,
        metric="euclidean",
        backend="faiss",
        device=device,
        distributed="auto",
        return_per_sample=False,
    )

    eval_time = time.time() - start_eval

    if rank == 0:
        print(f"{'=' * 70}")
        print("Results:")
        print(f"  Neighborhood preservation: {np_score:.4f}")
        print(f"  Embedding time: {embed_time:.2f}s")
        print(f"  Evaluation time: {eval_time:.2f}s")
        print(f"  Total time: {embed_time + eval_time:.2f}s")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
