import gzip
import pickle
from io import BytesIO
import time
import requests
import os
import matplotlib.pyplot as plt
import numpy as np

import torch

# Import UMAP implementations
from torchdr import UMAP as TorchdrUMAP  # GPU-accelerated UMAP from torchdr
import umap  # Classic UMAP from umap-learn


def download_and_load_dataset(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(BytesIO(response.content), "rb") as f:
        data = pickle.load(f)
    return data


def load_datasets():
    # --- Download Macosko data ---
    url_macosko = "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz"
    data_macosko = download_and_load_dataset(url_macosko)
    x_macosko = data_macosko["pca_50"].astype("float32")

    # --- Download 10x mouse Zheng data ---
    url_10x = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
    data_10x = download_and_load_dataset(url_10x)
    x_10x = data_10x["pca_50"].astype("float32")

    return x_macosko, x_10x


def time_umap(model, X, device=None):
    """
    Fit the UMAP model on X and measure runtime.
    """
    start = time.perf_counter()
    embedding = model.fit_transform(X)
    elapsed = time.perf_counter() - start

    return elapsed, embedding


def plot_results(runtime_data, sample_counts):
    datasets = ["Single-cell : Macosko et al.", "Single-cell : 10x Mouse Zheng al. "]
    methods = ["UMAP (CPU)", "TorchDR UMAP (GPU)"]
    colors = ["#1f77b4", "#ff7f0e"]

    x = np.arange(len(methods)) * 1.5
    bar_width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, ds in zip(axes, datasets):
        ax.bar(x, runtime_data[ds], width=bar_width, color=colors)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=20, fontweight="bold", ha="center")

        ax.set_ylabel("Runtime (sec)", fontsize=18)

        ax.set_title(f"{ds} ({sample_counts[ds]:,} samples)", fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig("umap_benchmark.png", dpi=300)
    plt.show()


def main():
    x_macosko, x_10x = load_datasets()

    max_iter = 500
    kwargs_torchdr = {
        "max_iter": max_iter,
        "verbose": True,
        "backend": "faiss",
        "device": "cuda",
    }
    kwargs_umap = {
        "n_epochs": max_iter,
        "verbose": True,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("=== Macosko dataset ===")
    print(f"Number of samples: {x_macosko.shape[0]}")
    print(f"Number of features: {x_macosko.shape[1]}")

    classic_umap = umap.UMAP(**kwargs_umap)
    time_classic_macosko, emb_classic = time_umap(classic_umap, x_macosko)
    print(f"Classic UMAP runtime: {time_classic_macosko:.4f} seconds\n")

    torchdr_umap = TorchdrUMAP(**kwargs_torchdr)
    time_torchdr_macosko, emb_torchdr = time_umap(
        torchdr_umap, x_macosko, device=device
    )
    print(f"Torchdr UMAP runtime: {time_torchdr_macosko:.4f} seconds")

    print("=== 10x Mouse Zheng dataset ===")
    print(f"Number of samples: {x_10x.shape[0]}")
    print(f"Number of features: {x_10x.shape[1]}")

    classic_umap = umap.UMAP(**kwargs_umap)
    time_classic_10x, emb_classic = time_umap(classic_umap, x_10x)
    print(f"Classic UMAP runtime: {time_classic_10x:.4f} seconds")

    torchdr_umap = TorchdrUMAP(**kwargs_torchdr)
    time_torchdr_10x, emb_torchdr = time_umap(torchdr_umap, x_10x, device=device)
    print(f"Torchdr UMAP runtime: {time_torchdr_10x:.4f} seconds")

    runtime_data = {
        "Macosko": [time_classic_macosko, time_torchdr_macosko],
        "10x Mouse Zheng": [time_classic_10x, time_torchdr_10x],
    }

    sample_counts = {"Macosko": x_macosko.shape[0], "10x Mouse Zheng": x_10x.shape[0]}

    plot_results(runtime_data, sample_counts)


if __name__ == "__main__":
    main()
