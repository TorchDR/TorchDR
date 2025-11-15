"""Multi-GPU k-NN label accuracy benchmark.

This example demonstrates how to use TorchDR's k-NN label accuracy metric
in distributed mode across multiple GPUs. The metric measures how well
local class structure is preserved in the data representation.

Dataset: Zheng et al. 2017 - 10x Mouse (1.3M single cells)
Metric: k-NN label accuracy (proportion of k-nearest neighbors with same label)

Usage:
    # Single GPU
    python knn_accuracy_benchmark.py

    # Multi-GPU (2 GPUs)
    torchdr run --gpus 2 knn_accuracy_benchmark.py

    # Multi-GPU (all available GPUs)
    torchdr run knn_accuracy_benchmark.py
"""

import time
import gzip
import pickle
from io import BytesIO

import requests
import torch
from sklearn.preprocessing import LabelEncoder

from torchdr.eval import knn_label_accuracy
from torchdr.distributed import is_distributed, get_rank, get_world_size


def load_zheng_dataset():
    """Load Zheng et al. 2017 10x mouse dataset (1.3M cells, 50 PCA dims)."""
    url = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with gzip.open(BytesIO(response.content), "rb") as f:
        data = pickle.load(f)

    X = torch.from_numpy(data["pca_50"].astype("float32"))
    y = data["CellType1"]
    y_encoded = torch.from_numpy(LabelEncoder().fit_transform(y))

    return X, y_encoded


def main():
    rank = get_rank()
    world_size = get_world_size()

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("k-NN Label Accuracy Benchmark")
        print("Dataset: Zheng et al. 2017 (10x Mouse, 1.3M cells)")
        print(f"Configuration: {world_size} GPU{'s' if world_size > 1 else ''}")
        print(f"{'=' * 70}\n")

    if rank == 0:
        print("Loading dataset...")

    data, labels = load_zheng_dataset()
    n_samples, n_features = data.shape

    if rank == 0:
        print(f"  Samples: {n_samples:,}")
        print(f"  Features: {n_features}")
        print(f"  Classes: {len(labels.unique())}\n")

    k = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"Computing k-NN label accuracy (k={k})...")
        print("  Backend: FAISS")
        print(f"  Device: {device}")
        print(f"  Distributed: {'Yes' if is_distributed() else 'No'}\n")

    start_time = time.time()

    accuracy = knn_label_accuracy(
        data,
        labels,
        k=k,
        metric="euclidean",
        backend="faiss",
        exclude_self=True,
        distributed="auto",
        return_per_sample=False,
        device=device,
    )

    elapsed_time = time.time() - start_time

    if rank == 0:
        print(f"{'=' * 70}")
        print("Results:")
        print(f"  k-NN accuracy: {accuracy:.4f}")
        print(f"  Total time: {elapsed_time:.2f}s")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
