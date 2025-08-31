"""
Example of using LargeVisMultiGPU with Tahoe-100M SCVI embeddings.
Must be launched with torchrun for distributed execution.

Usage:
    torchrun --nproc_per_node=8 tahoe_scvi_largevis_multigpu.py
"""

import os
import torch
import torch.distributed as dist
import scanpy as sc
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from torchdr.neighbor_embedding.largevis_multi_gpu import LargeVisMultiGPU


def setup_distributed():
    """Initialize distributed training environment."""
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
        print("Loading Tahoe-100M AnnData with pre-computed SCVI embeddings...")

    # Path to the downloaded SCVI model and data
    model_dir = "/braid/vanasseh/tahoe-100m-scvi-v1"
    adata_path = os.path.join(model_dir, "adata.h5ad")

    # Only rank 0 loads the AnnData object to save memory
    if rank == 0:
        print(f"Loading AnnData from {adata_path}...")
        print("This may take a few minutes due to the large file size (40GB)...")
        print("Only loading on rank 0 to save memory...")

        adata = sc.read_h5ad(adata_path)

        print(f"Loaded AnnData with {adata.n_obs:,} cells and {adata.n_vars:,} genes")
        print(f"Available metadata: {list(adata.obs.columns)}")
        print(f"Available embeddings in obsm: {list(adata.obsm.keys())}")

        # The SCVI embeddings are already pre-computed in the AnnData object
        # X_latent_qzm is the mean of the latent distribution (10-dimensional)
        if "X_latent_qzm" in adata.obsm:
            latent = adata.obsm["X_latent_qzm"]
        elif "_scvi_latent_qzm" in adata.obsm:
            latent = adata.obsm["_scvi_latent_qzm"]
        else:
            raise ValueError(
                "SCVI latent embeddings not found in adata.obsm. Available keys: "
                + str(list(adata.obsm.keys()))
            )

        n_obs = adata.n_obs
    else:
        latent = None
        n_obs = None
        adata = None

    # Broadcast the number of observations to all ranks
    n_obs_list = [n_obs]
    dist.broadcast_object_list(n_obs_list, src=0)
    n_obs = n_obs_list[0]

    # Sample a subset if full dataset is too large
    # You can adjust this number based on available memory
    n_samples = min(10000000, n_obs)  # Start with 1M cells for testing

    if rank == 0:
        print(f"Using pre-computed SCVI embeddings with shape: {latent.shape}")
        print(f"Embedding dimensions: {latent.shape[1]} (should be 10)")

        if n_samples < n_obs:
            print(f"Using a subset of {n_samples:,} cells for LargeVis computation")
            # Random sampling using torch
            indices = torch.randperm(n_obs)[:n_samples]
            latent = latent[indices.numpy()]

        # Convert to torch tensor (stays on CPU)
        x_tensor = torch.from_numpy(latent).float()
        tensor_shape = x_tensor.shape
    else:
        x_tensor = None
        tensor_shape = None

    # Broadcast tensor shape to all ranks
    shape_list = [tensor_shape]
    dist.broadcast_object_list(shape_list, src=0)
    tensor_shape = shape_list[0]

    # Allocate tensor on all ranks and broadcast the data
    # NCCL requires GPU tensors for broadcast, so we temporarily move to GPU
    if rank != 0:
        x_tensor = torch.empty(tensor_shape, dtype=torch.float32)

    # Move to GPU for broadcast
    x_tensor_gpu = x_tensor.cuda()
    dist.broadcast(x_tensor_gpu, src=0)

    # Move back to CPU (as LargeVis expects CPU input)
    x_tensor = x_tensor_gpu.cpu()

    if rank == 0:
        print(f"SCVI embeddings shape: {x_tensor.shape}")
        print(f"Computing LargeVis embedding with {world_size} GPUs...")

    # No labels needed - we'll plot without colors

    # Synchronize all ranks before starting computation
    dist.barrier()

    # Configure LargeVis with parameters suitable for large-scale data
    largevis = LargeVisMultiGPU(
        perplexity=30,  # Can be increased for larger datasets
        n_components=2,
        lr="auto",
        max_iter=1000,  # Reduced iterations for initial testing
        verbose=(rank == 0),  # Only rank 0 prints progress
        random_state=42,
        n_negatives=5,
        sparsity=True,
        compile=True,  # Enable for potential speedup
        init="pca",  # PCA now properly handles CPU/GPU transfer
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

    # Verify all GPUs have the same embedding
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

        # Plot the embedding
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Subsample for plotting if too many points
        plot_n = min(50000, len(z_largevis_np))
        if plot_n < len(z_largevis_np):
            plot_indices = torch.randperm(len(z_largevis_np))[:plot_n].numpy()
            z_plot = z_largevis_np[plot_indices]
            print(f"Plotting {plot_n:,} points out of {len(z_largevis_np):,}")
        else:
            z_plot = z_largevis_np

        scatter = ax.scatter(
            z_plot[:, 0],
            z_plot[:, 1],
            c="blue",
            s=0.5,
            alpha=0.6,
            rasterized=True,  # Rasterize for large datasets
        )

        title = f"LargeVis Multi-GPU ({world_size} GPUs) - Tahoe-100M SCVI\n"
        title += f"Cells: {n_samples:,}"
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("LargeVis Component 1")
        ax.set_ylabel("LargeVis Component 2")

        # No legend needed without labels

        # Save the figure
        output_filename = (
            f"tahoe_scvi_largevis_multigpu_{world_size}gpus_{n_samples}cells.png"
        )
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved as {output_filename}")

        # Also save the embeddings as torch tensor
        torch_output_filename = (
            f"tahoe_scvi_largevis_embeddings_{world_size}gpus_{n_samples}cells.pt"
        )
        torch.save(torch.from_numpy(z_largevis_np), torch_output_filename)
        print(f"Embeddings saved as {torch_output_filename}")

        plt.show()

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()
