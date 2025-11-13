"""Test distributed UMAP on MNIST with 2 GPUs.

Launch with: torchrun --nproc_per_node=2 test_distributed_umap.py
"""

import torch.distributed as dist
from sklearn.datasets import fetch_openml
from torchdr import UMAP

# Initialize distributed
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

if rank == 0:
    print(f"Running distributed UMAP on {world_size} GPUs")
    print("Loading MNIST...")

# Load MNIST
X = fetch_openml("mnist_784", parser="auto").data.astype("float32")

if rank == 0:
    print(f"Data shape: {X.shape}")
    print("Running UMAP with distributed=True...")

# Run UMAP with distributed mode - use defaults, just specify cuda and distributed
umap = UMAP(
    device="cuda",
    distributed=True,
    verbose=True,
)

X_embedded = umap.fit_transform(X)

if rank == 0:
    print(f"Embedding shape on rank 0: {X_embedded.shape}")
    print(f"Expected shape: approximately {X.shape[0] // world_size} rows")
    print("Distributed UMAP test completed successfully!")

# Cleanup
dist.destroy_process_group()
