from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from torchdr import PCA, TSNE, UMAP, LargeVis, InfoTSNE

# --- Load the MNIST dataset ---
mnist = fetch_openml("mnist_784", cache=True, as_frame=False)

x = mnist.data.astype("float32")
y = mnist.target.astype("int64")

# --- Perform PCA as a preprocessing step ---
x = PCA(50).fit_transform(x)

# --- Compute LargeVis embedding ---
largevis = LargeVis(
    keops=True,
    device="cuda",
    verbose=True,
    max_iter=5000,
)
z_largevis = largevis.fit_transform(x)

# # --- Compute UMAP embedding ---
# umap = UMAP(keops=True, device="cuda", verbose=True)
# z_umap = umap.fit_transform(x)

plt.scatter(z_largevis[:, 0], z_largevis[:, 1], c=y, cmap="tab10", s=1, alpha=0.5)
plt.title("LargeVis")

plt.savefig("test.png", format="png", bbox_inches="tight")

plt.show()
