"""Example of using FAISS configuration options in TorchDR.

This example demonstrates how to use various FAISS configuration options
for efficient k-NN computation in TorchDR, including float16 support,
multi-GPU setup, and memory management.
"""

import torch
import numpy as np
from torchdr import UMAP, TSNE
from torchdr.distance import FaissConfig
from torchdr.affinity import UMAPAffinity, EntropicAffinity
from sklearn.datasets import load_digits

# Load example data
X, y = load_digits(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32)

print("Dataset shape:", X.shape)
print("Available device:", "cuda" if torch.cuda.is_available() else "cpu")

# Example 1: Basic FAISS with float16 for memory efficiency
# -----------------------------------------------------------
print("\n1. Basic float16 configuration:")

if torch.cuda.is_available():
    X_gpu = X.cuda()

    # Configure FAISS with float16
    faiss_config = FaissConfig(use_float16=True)

    # Use with UMAP
    umap = UMAP(
        n_neighbors=15,
        backend="faiss",
        backend_config=faiss_config,
        device="cuda",
        verbose=True,
    )

    embedding = umap.fit_transform(X_gpu)
    print(f"UMAP embedding shape: {embedding.shape}")
    print(f"Using float16 reduces GPU memory by ~50%")

# Example 2: Custom memory allocation
# ------------------------------------
print("\n2. Custom memory configuration:")

if torch.cuda.is_available():
    # Configure with 2GB temporary memory
    faiss_config = FaissConfig(
        use_float16=True,
        temp_memory=2.0,  # 2GB
    )

    # Use with t-SNE
    tsne = TSNE(
        perplexity=30,
        backend="faiss",
        backend_config=faiss_config,
        device="cuda",
        verbose=True,
    )

    embedding = tsne.fit_transform(X_gpu)
    print(f"t-SNE embedding shape: {embedding.shape}")

# Example 3: Using dictionary configuration
# ------------------------------------------
print("\n3. Dictionary configuration (alternative syntax):")

if torch.cuda.is_available():
    # Can also pass configuration as a dictionary
    umap = UMAP(
        n_neighbors=15,
        backend="faiss",
        backend_config={"use_float16": True, "temp_memory": 1.5},
        device="cuda",
    )

    embedding = umap.fit_transform(X_gpu)
    print(f"UMAP embedding shape: {embedding.shape}")

# Example 4: Direct affinity usage with FAISS config
# ---------------------------------------------------
print("\n4. Direct affinity computation with FAISS:")

if torch.cuda.is_available():
    # Create FAISS configuration
    faiss_config = FaissConfig(use_float16=True)

    # Use with UMAPAffinity
    affinity = UMAPAffinity(
        n_neighbors=15, backend="faiss", backend_config=faiss_config, device="cuda"
    )

    # Compute affinity matrix
    P = affinity(X_gpu)
    print(f"Affinity matrix computed with FAISS backend")

# Example 5: Memory-constrained environments
# -------------------------------------------
print("\n5. Memory-constrained configuration:")

if torch.cuda.is_available():
    # Minimal memory allocation
    faiss_config = FaissConfig(
        use_float16=True,
        temp_memory=0.5,  # Only 512MB
    )

    # Or disable pre-allocation entirely
    faiss_config_minimal = FaissConfig(
        use_float16=True,
        temp_memory=0,  # Use cudaMalloc on demand
    )

    print("Configured for memory-constrained environment")

# Example 6: Multi-GPU setup (if available)
# ------------------------------------------
print("\n6. Multi-GPU configuration:")

if torch.cuda.device_count() > 1:
    print(f"Found {torch.cuda.device_count()} GPUs")

    # Configure for multiple GPUs with sharding
    faiss_config = FaissConfig(
        device=[0, 1],  # Use first two GPUs
        shard=True,  # Split dataset across GPUs
        use_float16=True,
    )

    umap = UMAP(
        n_neighbors=15, backend="faiss", backend_config=faiss_config, device="cuda"
    )

    # For large datasets, this would split the data across GPUs
    embedding = umap.fit_transform(X_gpu)
    print(f"UMAP embedding with multi-GPU: {embedding.shape}")
else:
    print("Multi-GPU not available on this system")

# Example 7: Guidelines for configuration
# ----------------------------------------
print("\n7. Configuration guidelines:")
print("""
Recommended settings based on dataset size:
- Small datasets (<10K points): Default settings are fine
- Medium datasets (10K-100K points): use_float16=True
- Large datasets (100K-1M points): use_float16=True, temp_memory=2-4GB
- Very large datasets (>1M points): Multi-GPU with sharding

Memory settings:
- temp_memory='auto': Use FAISS default (~18% of GPU memory)
- temp_memory=2.0: Allocate 2GB for temporary computations
- temp_memory=0: Disable pre-allocation (use for debugging)

Float16 benefits:
- Reduces memory usage by ~50%
- Often improves performance on modern GPUs
- Minimal impact on accuracy for most DR tasks
""")
