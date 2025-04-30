#!/usr/bin/env python3
"""Example of DistR on MNIST dataset."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

import torch
from torchdr.affinity import (
    EntropicAffinity,
    StudentAffinity,
    GaussianAffinity,
    ScalarProductAffinity,
)
from torchdr import DistR, PCA

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST dataset
mnist = fetch_openml("mnist_784", cache=True, as_frame=False)
y = mnist.target.astype(int)

# Preprocess data
X = PCA(50).fit_transform(mnist.data)
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

# Configure affinities
# affinity_in = EntropicAffinity(perplexity=30)
# affinity_in = GaussianAffinity()
# affinity_out = StudentAffinity()
affinity_in = ScalarProductAffinity()
affinity_out = ScalarProductAffinity()

# Create and fit DistR model
distr = DistR(
    affinity_in=affinity_in,
    affinity_out=affinity_out,
    n_components=2,
    loss_fn="square_loss",
    n_prototypes=50,  # Using 50 prototypes
    n_iter_mirror_descent=5,
    epsilon_mirror_descent=1e2,
    max_iter=300,
    init="random",
    init_scaling=1e-4,
    verbose=True,
    device=device,
    lr=1e-3,
)

# Fit model
embedding = distr.fit_transform(X_tensor)

# Plot results
plt.figure(figsize=(10, 8))

# Move embeddings to CPU for plotting
embedding_cpu = embedding.cpu().numpy() if torch.is_tensor(embedding) else embedding

# Plot data points
plt.scatter(
    embedding_cpu[:, 0],
    embedding_cpu[:, 1],
    c=y,
    cmap="tab10",
    alpha=0.6,
    s=10,
)

# Highlight prototypes differently
prototype_embeddings = distr.embedding_
prototype_embeddings_cpu = prototype_embeddings.cpu().numpy()
plt.scatter(
    prototype_embeddings_cpu[:, 0],
    prototype_embeddings_cpu[:, 1],
    marker="*",
    s=150,
    c="red",
    edgecolor="black",
    label="Prototypes",
)

plt.colorbar(label="Digit")
plt.title("DistR embeddings of MNIST digits with 50 prototypes")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(alpha=0.3)

# Visualize transport plan for a subset of points
plt.figure(figsize=(10, 8))
plt.title("Transport Plan Visualization (partial)")
plt.imshow(distr.OT_plan_[:100, :].cpu().numpy(), aspect="auto", cmap="viridis")
plt.colorbar(label="Transport Weight")
plt.xlabel("Prototype Index")
plt.ylabel("Data Point Index")

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Number of data points: {X.shape[0]}")
print(f"Number of prototypes: {distr.n_prototypes}")
print(
    f"Final loss: {distr.Loss(distr.PX_, distr.affinity_out(distr.embedding_), torch.ones(distr.n_samples_in_, device=distr.device), distr.OT_plan_.sum(dim=0), distr.OT_plan_).item()}"
)
