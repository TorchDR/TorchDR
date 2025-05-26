#!/usr/bin/env python3
"""Example of DistR on MNIST dataset."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from collections import Counter

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
affinity_in = EntropicAffinity(perplexity=100, backend="faiss", sparsity=True)
# affinity_in = GaussianAffinity()
affinity_out = StudentAffinity()

# Create and fit DistR model
distr = DistR(
    affinity_in=affinity_in,
    affinity_out=affinity_out,
    n_components=2,
    loss_fn="kl_loss",
    n_prototypes=50,  # Using 50 prototypes
    n_iter_mirror_descent=5,
    lr_mirror_descent=1e-4,
    max_iter=300,
    init="random",
    init_scaling=1e-4,
    verbose=True,
    device=device,
    lr=1e-1,
)

# Fit model
embedding = distr.fit_transform(X_tensor)

# Move embeddings to CPU for plotting
embedding_cpu = embedding.cpu().numpy() if torch.is_tensor(embedding) else embedding

# Get the OT plan and move it to CPU for analysis
ot_plan = distr.OT_plan_.cpu().numpy()

# For each data point, get the prototype it's most associated with
point_to_prototype = np.argmax(ot_plan, axis=1)

# For each prototype, find the best represented label
prototype_labels = {}
for prototype_idx in range(distr.n_prototypes):
    # Get indices of points assigned to this prototype
    point_indices = np.where(point_to_prototype == prototype_idx)[0]

    if len(point_indices) > 0:
        # Get labels of these points
        cluster_labels = y[point_indices]

        # Count occurrences and find the most common label
        label_counts = Counter(cluster_labels)
        dominant_label = label_counts.most_common(1)[0][0]

        prototype_labels[prototype_idx] = dominant_label

# Plot results
plt.figure(figsize=(10, 8))

# Get prototype embeddings
prototype_embeddings = distr.embedding_
prototype_embeddings_cpu = prototype_embeddings.detach().cpu().numpy()

# Create a color array for the prototypes
prototype_colors = [prototype_labels.get(i, -1) for i in range(distr.n_prototypes)]

# Plot only the prototypes, colored by their dominant label
plt.scatter(
    prototype_embeddings_cpu[:, 0],
    prototype_embeddings_cpu[:, 1],
    marker="o",
    s=80,
    c=prototype_colors,
    cmap="tab10",
    edgecolor="black",
)

# Add labels to each prototype showing its number and dominant digit
for i, (x, y) in enumerate(prototype_embeddings_cpu):
    label = prototype_labels.get(i, -1)
    plt.annotate(
        f"{i}:{label}",
        (x, y),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

# Add color legend
plt.colorbar(label="Dominant Digit")
plt.title("DistR prototypes colored by their dominant digit label")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mnist_distr_prototypes.png")
plt.show()

# Print some statistics
print(f"Number of data points: {X.shape[0]}")
print(f"Number of prototypes: {distr.n_prototypes}")
# Print the dominant label for each prototype
for proto_idx, label in sorted(prototype_labels.items()):
    print(f"Prototype {proto_idx}: Dominant label = {label}")
