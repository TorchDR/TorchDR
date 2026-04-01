r"""
Non-Parametric UMAP Transform on Handwritten Digits
==================================================

We fit a reference UMAP embedding on one subset of the sklearn handwritten
digits dataset and then map unseen samples with :meth:`torchdr.UMAP.transform`
instead of refitting on the full dataset. This mimics a deployment workflow
where a reference embedding is already available and new samples only arrive
with their input features.

We use the built-in digits dataset rather than full MNIST so the example stays
fully self-contained and lightweight enough for the documentation gallery. By
default, we fit the reference embedding on 1,400 points and transform 300
inference-only query points. We report a simple deployment metric: 10-NN label
transfer accuracy from the reference embedding to the transformed query points.

"""

# Author: OpenAI Codex
#
# License: BSD 3-Clause License

import os

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from torchdr import UMAP


RANDOM_STATE = 0
N_REFERENCE = int(os.environ.get("TORCHDR_EXAMPLE_N_REFERENCE", "1400"))
N_QUERY = int(os.environ.get("TORCHDR_EXAMPLE_N_QUERY", "300"))
PCA_DIM = 50
MAX_ITER = int(os.environ.get("TORCHDR_EXAMPLE_MAX_ITER", "150"))


# %%
# Load handwritten digits and prepare a reference/query split
# -----------------------------------------------------------
#
# We keep a reference pool that is used during fitting and a disjoint query pool
# that is embedded later only through :meth:`transform`.

digits = load_digits()
X = (digits.data / 16.0).astype("float32")
y = digits.target.astype("int64")

if N_REFERENCE + N_QUERY > len(X):
    raise ValueError(
        f"Requested {N_REFERENCE + N_QUERY} samples but digits only contains {len(X)}."
    )

X_reference, X_query, y_reference, y_query = train_test_split(
    X,
    y,
    train_size=N_REFERENCE,
    test_size=N_QUERY,
    stratify=y,
    random_state=RANDOM_STATE,
)


# %%
# Preprocess features and fit the reference embedding
# ---------------------------------------------------
#
# We reduce the input features with PCA before fitting UMAP, then transform the
# held-out query set with the non-parametric transform path.

pca = PCA(n_components=PCA_DIM)
X_reference_pca = pca.fit_transform(X_reference).astype("float32")
X_query_pca = pca.transform(X_query).astype("float32")

umap = UMAP(
    n_components=2,
    n_neighbors=15,
    max_iter=MAX_ITER,
    init="pca",
    optimizer="SGD",
    backend=None,
    device="cpu",
    random_state=RANDOM_STATE,
)

Z_reference = umap.fit_transform(X_reference_pca)
Z_query = umap.transform(X_query_pca, X_train=X_reference_pca)


# %%
# Evaluate deployment-time label transfer
# ---------------------------------------
#
# In a production setting, a simple use case is to classify or annotate new
# points by comparing them to the already embedded reference set.

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(Z_reference, y_reference)
label_transfer_accuracy = knn.score(Z_query, y_query)

print(f"Reference samples used for fit: {len(X_reference_pca)}")
print(f"Query samples used for transform only: {len(X_query_pca)}")
print(f"10-NN label transfer accuracy: {label_transfer_accuracy:.3f}")


# %%
# Visualize the reference and transformed query points
# ----------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    Z_reference[:, 0],
    Z_reference[:, 1],
    c=y_reference,
    cmap="tab10",
    s=5,
    alpha=0.55,
)
axes[0].set_title(f"Reference embedding\nfit on {len(X_reference_pca)} points")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].scatter(
    Z_reference[:, 0],
    Z_reference[:, 1],
    c="lightgray",
    s=3,
    alpha=0.12,
)
scatter = axes[1].scatter(
    Z_query[:, 0],
    Z_query[:, 1],
    c=y_query,
    cmap="tab10",
    s=7,
    alpha=0.8,
)
axes[1].set_title(
    "Inference-only query points\n"
    f"transform on {len(X_query_pca)} points, "
    f"10-NN accuracy = {label_transfer_accuracy:.3f}"
)
axes[1].set_xticks([])
axes[1].set_yticks([])

handles, labels = scatter.legend_elements(prop="colors")
fig.legend(handles, labels, loc="lower center", ncol=10, frameon=False)
plt.subplots_adjust(bottom=0.16, wspace=0.08)
plt.show()
