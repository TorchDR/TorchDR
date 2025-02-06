r"""
Incremental PCA on GPU
======================

This example demonstrates how to use the `IncrementalPCA` class on GPU.
We compare the memory usage and time taken to fit the model with the regular
`PCA` class on GPU.
"""

import time
import gc
import torch

from torchdr import IncrementalPCA, PCA

# Choose the GPU device if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("This example requires a CUDA-capable GPU.")

# Generate a large random dataset on CPU.
n_samples = 100_000
n_features = 500
X = torch.randn(n_samples, n_features)  # Stored on CPU

# --------------------------
# Incremental PCA on GPU
# --------------------------
# IncrementalPCA is designed so that only each batch is moved to the GPU.
# Set n_components=50 and choose an appropriate batch_size.
ipca = IncrementalPCA(n_components=50, batch_size=1024, device=device)

# Reset GPU memory stats before fitting.
torch.cuda.reset_peak_memory_stats(device)

start = time.time()
ipca.fit(X, check_input=True)
ipca_time = time.time() - start

# Get the peak GPU memory allocated (in bytes).
ipca_peak_mem = torch.cuda.max_memory_allocated(device)
print(
    f"Incremental PCA on GPU:\n"
    f"  Time: {ipca_time:.2f} sec, "
    f"Peak GPU memory: {ipca_peak_mem / 1024**2:.2f} MB"
)

# Clean up and free GPU memory.
del ipca
torch.cuda.empty_cache()
gc.collect()

# --------------------------
# Regular PCA on GPU
# --------------------------
# Regular PCA (from torchdr) will typically move the full dataset to GPU.
pca = PCA(n_components=50, device=device)

# Reset GPU memory stats before fitting.
torch.cuda.reset_peak_memory_stats(device)

start = time.time()
# Here we assume that pca.fit will move the full dataset to the GPU.
pca.fit(X)
pca_time = time.time() - start

# Get the peak GPU memory allocated (in bytes).
pca_peak_mem = torch.cuda.max_memory_allocated(device)
print(
    f"\nRegular PCA on GPU:\n"
    f"  Time: {pca_time:.2f} sec, "
    f"Peak GPU memory: {pca_peak_mem / 1024**2:.2f} MB"
)
