r"""
Comparison of different DR methods and the use of affinity matcher
==================================================================

We illustrate the basic usage of TorchDR with different Neighbor Embedding methods
on the swiss roll dataset.

"""

# %%
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

from torchdr import (
    AffinityMatcher,
    SNE,
    UMAP,
    TSNE,
    EntropicAffinity,
    NormalizedGaussianAffinity,
)

# %%
# Load toy images
# ---------------
#
# First, let's load 5 classes of the digits dataset from sklearn.
torch.manual_seed(0)
n_samples = 500
X, t = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=0)

init_embedding = torch.normal(0, 1, size=(n_samples, 2), dtype=torch.double)
# %%
# Compute the different embedding
# -------------------------------
#
# Tune the different hyperparameters for better results.
perplexity = 30
lr = 1e-1
optim_params = {
    "init": init_embedding,
    "early_exaggeration_iter": 0,
    "optimizer": "Adam",
    "optimizer_kwargs": None,
    "early_exaggeration": 1.0,
    "max_iter": 100,
}

sne = SNE(n_components=2, perplexity=perplexity, lr=lr, **optim_params)

umap = UMAP(n_neighbors=perplexity, n_components=2, lr=lr, **optim_params)

tsne = TSNE(n_components=2, perplexity=perplexity, lr=lr, **optim_params)

all_methods = {
    "TSNE": tsne,
    "SNE": sne,
    "UMAP": umap,
}

for method_name, method in all_methods.items():
    print("--- Computing {} ---".format(method_name))
    method.fit(X)

# %%
# Plot the different embeddings
# -----------------------------
fig = plt.figure(figsize=(15, 4))
fs = 24
ax = fig.add_subplot(1, 4, 1, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, s=20)
ax.set_title("Swiss Roll in ambient space", font="Times New Roman", fontsize=fs)
ax.view_init(azim=-66, elev=12)

for i, (method_name, method) in enumerate(all_methods.items()):
    ax = fig.add_subplot(1, 4, i + 2)
    emb = method.embedding_.detach().numpy()  # get the embedding
    ax.scatter(emb[:, 0], emb[:, 1], c=t, s=20)
    ax.set_title("{0}".format(method_name), font="Times New Roman", fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
# %%
# Using AffinityMatcher
# -----------------------------
#
# We can reproduce the same kind of results using the
# flexible class AffinityMatcher
# :class:`torchdr.AffinityMatcher`. It take as input
# two affinities and minimize a certain matching loss
# between them. To reproduce the SNE algorithm
# we can match with the cross entropy loss
# an EntropicAffinity
# :class:`torchdr.EntropicAffinity` with given
# perplexity and a NormalizedGaussianAffinity
# :class:`torchdr.NormalizedGaussianAffinity`.

sne_affinity_matcher = AffinityMatcher(
    n_components=2,
    # SNE matches an EntropicAffinity
    affinity_in=EntropicAffinity(perplexity=perplexity),
    # with a Gaussian kernel normalized by row
    affinity_out=NormalizedGaussianAffinity(normalization_dim=1),
    loss_fn="cross_entropy_loss",  # and the cross_entropy loss
    init=init_embedding,
    max_iter=200,
    lr=lr,
)
sne_affinity_matcher.fit(X)

fig = plt.figure(figsize=(10, 4))
fs = 24
two_sne_dict = {"SNE": sne, "SNE (with affinity matcher)": sne_affinity_matcher}
for i, (method_name, method) in enumerate(two_sne_dict.items()):
    ax = fig.add_subplot(1, 2, i + 1)
    emb = method.embedding_.detach().numpy()  # get the embedding
    ax.scatter(emb[:, 0], emb[:, 1], c=t, s=20)
    ax.set_title("{0}".format(method_name), font="Times New Roman", fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
