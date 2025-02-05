import os

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from torchdr import PCA, TSNE, UMAP, InfoTSNE, LargeVis

# --- Load the MNIST dataset ---
mnist = fetch_openml("mnist_784", cache=True, as_frame=False)

x = mnist.data.astype("float32")
y = mnist.target.astype("int64")

# --- Perform PCA as a preprocessing step ---
x = PCA(50).fit_transform(x)

# --- Compute InfoTSNE embedding ---
infotsne = InfoTSNE(backend="faiss", device="cuda", verbose=True)
z_infotsne = infotsne.fit_transform(x)
