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
    torchrun --nproc_per_node=2 knn_accuracy_benchmark.py

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 knn_accuracy_benchmark.py
"""

import os
import time
import gzip
import pickle
from io import BytesIO

import requests
import torch
import torch.distributed as dist
from sklearn.preprocessing import LabelEncoder

from torchdr.eval import knn_label_accuracy


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
    y = data["CellType1"].astype("str")
    y_encoded = torch.from_numpy(LabelEncoder().fit_transform(y))

    return X, y_encoded


def main():
    is_distributed, rank, world_size = setup_distributed()

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

    if is_distributed:
        dist.barrier()

    k = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"Computing k-NN label accuracy (k={k})...")
        print("  Backend: FAISS")
        print(f"  Device: {device}")
        print(f"  Distributed: {'Yes' if is_distributed else 'No'}\n")

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

    if is_distributed:
        all_accuracies = [None] * world_size
        all_times = [None] * world_size

        accuracy_cpu = accuracy.cpu().item() if torch.is_tensor(accuracy) else accuracy

        dist.gather_object(accuracy_cpu, all_accuracies if rank == 0 else None, dst=0)
        dist.gather_object(elapsed_time, all_times if rank == 0 else None, dst=0)

        if rank == 0:
            global_accuracy = sum(all_accuracies) / len(all_accuracies)
            max_time = max(all_times)

            print(f"{'=' * 70}")
            print("Results:")
            print(f"  k-NN accuracy: {global_accuracy:.4f}")
            print(f"  Total time: {max_time:.2f}s")
            print("  Per-GPU times:")
            for gpu_rank, t in enumerate(all_times):
                print(f"    GPU {gpu_rank}: {t:.2f}s")
            print(f"{'=' * 70}\n")

        cleanup_distributed()
    else:
        print(f"{'=' * 70}")
        print("Results:")
        print(f"  k-NN accuracy: {accuracy:.4f}")
        print(f"  Total time: {elapsed_time:.2f}s")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
