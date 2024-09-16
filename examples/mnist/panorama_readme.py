import os

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from torchdr import PCA, TSNE, UMAP, LargeVis, InfoTSNE

# --- Load the MNIST dataset ---
mnist = fetch_openml("mnist_784", cache=True, as_frame=False)

x = mnist.data.astype("float32")
y = mnist.target.astype("int64")

# --- Perform PCA as a preprocessing step ---
x = PCA(50).fit_transform(x)

# --- Compute TSNE embedding ---
tsne = TSNE(keops=True, device="cuda", verbose=True)
z_tsne = tsne.fit_transform(x)

# --- Compute InfoTSNE embedding ---
infotsne = InfoTSNE(keops=True, device="cuda", verbose=True)
z_infotsne = infotsne.fit_transform(x)

# --- Compute LargeVis embedding ---
largevis = LargeVis(keops=True, device="cuda", verbose=True, max_iter=10000)
z_largevis = largevis.fit_transform(x)

# --- Compute UMAP embedding ---
umap = UMAP(keops=True, device="cuda", verbose=True, max_iter=10000)
z_umap = umap.fit_transform(x)


# --- Plot the embeddings ---
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fontsize = 25

scatter = axes[0].scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, cmap="tab10", s=1, alpha=0.5)
axes[0].set_title("TSNE", fontsize=fontsize)
axes[0].set_xticks([-10, 10])
axes[0].set_yticks([-10, 10])

axes[1].scatter(z_infotsne[:, 0], z_infotsne[:, 1], c=y, cmap="tab10", s=1, alpha=0.5)
axes[1].set_title("InfoTSNE", fontsize=fontsize)
axes[1].set_xticks([-10, 10])
axes[1].set_yticks([-10, 10])

axes[2].scatter(z_largevis[:, 0], z_largevis[:, 1], c=y, cmap="tab10", s=1, alpha=0.5)
axes[2].set_title("LargeVis", fontsize=fontsize)
axes[2].set_xticks([-5, 5])
axes[2].set_yticks([-5, 5])

axes[3].scatter(z_umap[:, 0], z_umap[:, 1], c=y, cmap="tab10", s=1, alpha=0.5)
axes[3].set_title("UMAP", fontsize=fontsize)
axes[3].set_xticks([-5, 5])
axes[3].set_yticks([-5, 5])

handles, labels = scatter.legend_elements(prop="colors", size=15)
legend_labels = [f"{i}" for i in range(10)]
fig.legend(handles, legend_labels, loc="lower center", ncol=10, fontsize=15)
plt.subplots_adjust(bottom=0.15, wspace=0.1)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "../../docs/source/figures/mnist_readme.png")
fig.savefig(save_path, format="png", bbox_inches="tight")

plt.show()
