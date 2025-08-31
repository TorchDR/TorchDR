"""
Example of using LargeVisMultiGPU with MNIST dataset.
Must be launched with torchrun for distributed execution.

Usage:
    torchrun --nproc_per_node=2 mnist_largevis_multigpu.py
"""

import os
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from torchdr import PCA
from torchdr.neighbor_embedding.largevis_multi_gpu import LargeVisMultiGPU
from torchdr.eval import silhouette_score


def setup_distributed():
    """Initialize distributed training environment."""
    # torchrun sets the environment variables, but we still need to init the process group
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Initialize distributed training
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Print from all ranks to verify all GPUs are active
    print(
        f"[Rank {rank}] Process started on GPU {torch.cuda.current_device()}, device name: {torch.cuda.get_device_name()}"
    )
    dist.barrier()  # Synchronize before continuing

    if rank == 0:
        print(f"\nRunning distributed LargeVis on {world_size} GPUs")
        print("Loading MNIST dataset...")

    # Load the MNIST dataset
    mnist = fetch_openml("mnist_784", cache=True, as_frame=False)
    x = mnist.data.astype("float32")
    y = mnist.target.astype("int64")

    # Perform PCA as a preprocessing step (same as in panorama_readme.py)
    if rank == 0:
        print("Performing PCA preprocessing...")
    x = PCA(50).fit_transform(x)

    # Convert to torch tensor
    x_tensor = torch.from_numpy(x).cuda()

    # Compute LargeVis embedding using multi-GPU
    if rank == 0:
        print(f"Computing LargeVis embedding with {world_size} GPUs...")
        print(f"Data shape: {x_tensor.shape}")

    # Use similar parameters as the single GPU version but with multi-GPU
    largevis = LargeVisMultiGPU(
        perplexity=30,  # Default perplexity
        n_components=2,
        lr="auto",
        max_iter=1000,
        verbose=(rank == 0),  # Only rank 0 prints progress
        random_state=42,
        n_negatives=5,
        sparsity=True,
        compile=True,  # Can enable for potential speedup
        gradient_compression="fp16",
    )

    # Monitor GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated(rank)
        print(
            f"[Rank {rank}] GPU memory before training: {memory_before / 1024**3:.2f} GB"
        )

    # Fit and transform
    z_largevis = largevis.fit_transform(x_tensor)

    # Monitor GPU memory after training
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated(rank)
        print(
            f"[Rank {rank}] GPU memory after training: {memory_after / 1024**3:.2f} GB"
        )
        print(
            f"[Rank {rank}] Memory used for training: {(memory_after - memory_before) / 1024**3:.2f} GB"
        )

    # Verify each GPU processed its chunk
    print(
        f"[Rank {rank}] Embedding shape: {z_largevis.shape}, dtype: {z_largevis.dtype}"
    )

    # Convert back to numpy for plotting
    z_largevis_np = z_largevis.detach().cpu().numpy()

    # Synchronize all ranks before final evaluation
    dist.barrier()

    # Verify all GPUs have the same embedding (they should after all_reduce)
    embedding_norm = torch.norm(z_largevis).item()
    all_norms = [None] * world_size
    dist.all_gather_object(all_norms, embedding_norm)

    if rank == 0:
        print(f"\nEmbedding norms from all ranks (should be identical): {all_norms}")
        if len(set(all_norms)) == 1:
            print("✓ All GPUs have identical embeddings (as expected)")
        else:
            print("✗ WARNING: Embeddings differ across GPUs!")

    # Only rank 0 evaluates and plots the results
    if rank == 0:
        print("\nEmbedding complete! Shape:", z_largevis.shape)

        # Compute silhouette score on GPU
        print("Computing silhouette score on GPU...")
        y_tensor = torch.from_numpy(y).cuda()
        sil_score = silhouette_score(z_largevis, y_tensor)
        print(f"Silhouette Score: {sil_score:.4f}")

        # Plot the embedding
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        scatter = ax.scatter(
            z_largevis_np[:, 0], z_largevis_np[:, 1], c=y, cmap="tab10", s=1, alpha=0.5
        )
        ax.set_title(
            f"LargeVis Multi-GPU ({world_size} GPUs) - MNIST\nSilhouette Score: {sil_score:.4f}",
            fontsize=16,
        )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        # Add legend
        handles, labels = scatter.legend_elements(prop="colors")
        legend_labels = [f"{i}" for i in range(10)]
        ax.legend(handles, legend_labels, loc="best", title="Digit")

        # Save the figure
        output_filename = f"mnist_largevis_multigpu_{world_size}gpus.png"
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved as {output_filename}")

        plt.show()

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
