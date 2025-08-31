"""
Multi-GPU LargeVis on 10x Mouse Zheng single-cell dataset.
Run with: torchrun --nproc_per_node=2 benchmarks/largevis_10x_multigpu.py
"""

import gzip
import pickle
from io import BytesIO
import time
import requests
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from torchdr import LargeVis
from torchdr.neighbor_embedding.largevis_multi_gpu import LargeVisMultiGPU
from torchdr.eval import silhouette_score


def setup_distributed():
    """Initialize distributed training environment."""
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()


def download_and_load_dataset(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(BytesIO(response.content), "rb") as f:
        data = pickle.load(f)
    return data


def main():
    # Setup distributed training
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Running LargeVis with {world_size} GPUs on 10x Mouse Zheng dataset")
        print("=" * 60)

    # Download 10x mouse Zheng data
    if rank == 0:
        print("Downloading 10x Mouse Zheng dataset...")
    url_10x = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
    data_10x = download_and_load_dataset(url_10x)

    # Extract PCA-reduced data and labels
    x_10x = data_10x["pca_50"].astype("float32")
    y_10x = data_10x["CellType1"]

    # Convert labels to numeric
    unique_labels = np.unique(y_10x)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_to_idx[label] for label in y_10x])

    if rank == 0:
        print(f"Dataset shape: {x_10x.shape}")
        print(f"Number of samples: {x_10x.shape[0]}")
        print(f"Number of features: {x_10x.shape[1]}")
        print(f"Number of cell types: {len(unique_labels)}")
        print(f"Data range: [{x_10x.min():.3f}, {x_10x.max():.3f}]")
        print(f"Data std: {x_10x.std():.3f}")

    # Normalize data to prevent numerical issues
    x_10x = (x_10x - x_10x.mean()) / x_10x.std()

    if rank == 0:
        print(f"After normalization - mean: {x_10x.mean():.3f}, std: {x_10x.std():.3f}")
        print("=" * 60)

    # Convert to torch tensor
    x_tensor = torch.from_numpy(x_10x).cuda()
    y_tensor = torch.from_numpy(y_numeric).cuda()

    # Configure LargeVis
    max_iter = 500

    if rank == 0:
        print(f"Running LargeVis for {max_iter} iterations...")
        print(f"Each GPU processes {x_10x.shape[0] // world_size} samples")
        print("=" * 60)

    # Create multi-GPU LargeVis
    largevis_multi = LargeVisMultiGPU(
        perplexity=30,
        n_components=2,
        lr=1.0,  # Fixed learning rate
        max_iter=max_iter,
        verbose=(rank == 0),
        random_state=42,
        n_negatives=5,
        early_exaggeration_coeff=12.0,
        early_exaggeration_iter=250,
        check_interval=50,
    )

    # Time the embedding
    dist.barrier()
    torch.cuda.synchronize()
    start_time = time.time()

    z_multi = largevis_multi.fit_transform(x_tensor)

    torch.cuda.synchronize()
    multi_time = time.time() - start_time

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Multi-GPU LargeVis Results ({world_size} GPUs):")
        print(f"Total runtime: {multi_time:.2f} seconds")
        print(f"Time per iteration: {multi_time / max_iter:.3f} seconds")
        print(f"Embedding shape: {z_multi.shape}")

        # Check quality
        has_nan = torch.isnan(z_multi).any()
        print(f"Has NaN: {has_nan}")

        if not has_nan:
            print(f"Embedding range: [{z_multi.min():.3f}, {z_multi.max():.3f}]")

            # Compute silhouette score
            sil_score = silhouette_score(z_multi, y_tensor)
            print(f"Silhouette score: {sil_score:.4f}")

            # Save plot
            z_np = z_multi.cpu().numpy()

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            scatter = ax.scatter(
                z_np[:, 0], z_np[:, 1], c=y_numeric, cmap="tab20", s=0.5, alpha=0.5
            )
            ax.set_title(
                f"LargeVis Multi-GPU ({world_size} GPUs) - 10x Mouse Zheng\nSilhouette Score: {sil_score:.4f}"
            )
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            # Save the figure
            plt.savefig(
                "benchmarks/largevis_10x_multigpu.png", dpi=150, bbox_inches="tight"
            )
            print(f"Saved plot to benchmarks/largevis_10x_multigpu.png")

        # Compare with single GPU if world_size == 1
        if world_size == 1:
            print("\n" + "=" * 60)
            print("Running standard single-GPU LargeVis for comparison...")

            largevis_single = LargeVis(
                perplexity=30,
                n_components=2,
                lr=1.0,
                max_iter=max_iter,
                verbose=False,
                random_state=42,
                n_negatives=5,
                backend="faiss",
                device="cuda",
                early_exaggeration_coeff=12.0,
                early_exaggeration_iter=250,
            )

            torch.cuda.synchronize()
            start_time = time.time()

            z_single = largevis_single.fit_transform(x_tensor)

            torch.cuda.synchronize()
            single_time = time.time() - start_time

            print(f"Single-GPU LargeVis runtime: {single_time:.2f} seconds")
            print(
                f"Multi-GPU overhead: {(multi_time - single_time) / single_time * 100:.1f}%"
            )

        print("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
