import requests
import gzip
import pickle
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from torchdr import LargeVis, TSNE


def download_and_load_dataset(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(BytesIO(response.content), "rb") as f:
        data = pickle.load(f)
    return data


# --- Download Macosko data ---
url_macosko = "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz"
data_macosko = download_and_load_dataset(url_macosko)

x_macosko = data_macosko["pca_50"].astype("float32")
y_macosko = data_macosko["CellType1"].astype(str)
y_macosko_encoded = LabelEncoder().fit_transform(y_macosko)

# --- Download 10x mouse Zheng data ---
url_10x = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
data_10x = download_and_load_dataset(url_10x)

x_10x = data_10x["pca_50"].astype("float32")
y_10x = data_10x["CellType1"].astype("str")
y_10x_encoded = LabelEncoder().fit_transform(y_10x)


# --- Compute TSNE embeddings ---
tsne = TSNE(keops=True, device="cuda", verbose=True)
z_tsne = tsne.fit_transform(x_macosko)

# --- Compute LargeVis embeddings ---
largevis = LargeVis(
    verbose=True,
    device="cuda",
    keops=True,
    max_iter=1000,
    n_negatives=50,
    optimizer="Adam",
    lr=1e0,
    optimizer_kwargs=None,
)
z_largevis = largevis.fit_transform(x_10x)


# --- Plot embeddings ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fontsize = 15

scatter = axes[0].scatter(
    z_tsne[:, 0], z_tsne[:, 1], c=y_macosko_encoded, cmap="tab10", s=1, alpha=0.5
)
axes[0].set_title(
    r"TSNE on Macosko et al. 2015 $(4.5 \times 10^{4})$", fontsize=fontsize
)

axes[1].scatter(
    z_largevis[:, 0],
    z_largevis[:, 1],
    s=0.1,
    c=y_10x_encoded,
    alpha=0.5,
    cmap="gist_ncar",
)
axes[1].set_title(
    r"LargeVis on Zheng et al. 2017 $(1.3 \times 10^{6})$", fontsize=fontsize
)

plt.subplots_adjust(wspace=0.3)
fig.savefig("single_cell_readme.png", format="png", bbox_inches="tight")
plt.show()
