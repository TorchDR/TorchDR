import gzip
import pickle
from io import BytesIO
import time
import requests
import os

import torch
import psutil
from sklearn.preprocessing import LabelEncoder

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
    y_macosko = data_macosko["CellType1"].astype(str)
    y_macosko_encoded = LabelEncoder().fit_transform(y_macosko)

    # --- Download 10x mouse Zheng data ---
    url_10x = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"
    data_10x = download_and_load_dataset(url_10x)
    x_10x = data_10x["pca_50"].astype("float32")
    y_10x = data_10x["CellType1"].astype(str)
    y_10x_encoded = LabelEncoder().fit_transform(y_10x)

    return (x_macosko, y_macosko_encoded), (x_10x, y_10x_encoded)


def time_umap(model, X, device=None):
    """
    Fit the UMAP model on X and measure both runtime and memory usage.

    For torchdr UMAP (GPU-accelerated), we measure peak GPU memory usage.
    For classic UMAP (umap-learn), we measure the additional CPU memory used.
    """
    # For torchdr UMAP, reset GPU memory statistics if a CUDA device is used.
    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # For classic UMAP, measure CPU memory before fitting.
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    start = time.perf_counter()
    embedding = model.fit_transform(X)
    elapsed = time.perf_counter() - start

    if device is not None and device.type == "cuda":
        # Get peak GPU memory allocated in MB.
        mem_used = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        # For classic UMAP, measure the difference in CPU memory usage in MB.
        mem_after = process.memory_info().rss
        mem_used = (mem_after - mem_before) / (1024**2)

    return elapsed, embedding, mem_used


def main():
    # Load the datasets.
    (x_macosko, y_macosko), (x_10x, y_10x) = load_datasets()

    max_iter = 500
    kwargs_torchdr = {
        "max_iter": max_iter,
        "verbose": True,
        "backend": "keops",
        "device": "cuda",
    }
    kwargs_umap = {
        "n_epochs": max_iter,
        "verbose": True,
    }

    # Set device for torchdr UMAP: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Run on Macosko dataset ---
    print("=== Macosko dataset ===")
    print(f"Number of samples: {x_macosko.shape[0]}")
    print(f"Number of features: {x_macosko.shape[1]}")

    # Torchdr UMAP (GPU accelerated)
    torchdr_umap = TorchdrUMAP(**kwargs_torchdr)
    time_torchdr, emb_torchdr, mem_torchdr = time_umap(
        torchdr_umap, x_macosko, device=device
    )
    print(f"Torchdr UMAP runtime: {time_torchdr:.4f} seconds")
    print(f"Torchdr UMAP peak GPU memory usage: {mem_torchdr:.2f} MB")

    # Classic UMAP (umap-learn)
    classic_umap = umap.UMAP(**kwargs_umap)
    time_classic, emb_classic, mem_classic = time_umap(classic_umap, x_macosko)
    print(f"Classic UMAP runtime: {time_classic:.4f} seconds")
    print(f"Classic UMAP additional CPU memory usage: {mem_classic:.2f} MB\n")

    # --- Optionally: Run on 10x Mouse Zheng dataset ---
    print("=== 10x Mouse Zheng dataset ===")
    print(f"Number of samples: {x_10x.shape[0]}")
    print(f"Number of features: {x_10x.shape[1]}")

    # Torchdr UMAP on 10x dataset
    torchdr_umap = TorchdrUMAP(**kwargs_torchdr)
    time_torchdr, emb_torchdr, mem_torchdr = time_umap(
        torchdr_umap, x_10x, device=device
    )
    print(f"Torchdr UMAP runtime: {time_torchdr:.4f} seconds")
    print(f"Torchdr UMAP peak GPU memory usage: {mem_torchdr:.2f} MB")

    # Classic UMAP on 10x dataset
    classic_umap = umap.UMAP(**kwargs_umap)
    time_classic, emb_classic, mem_classic = time_umap(classic_umap, x_10x)
    print(f"Classic UMAP runtime: {time_classic:.4f} seconds")
    print(f"Classic UMAP additional CPU memory usage: {mem_classic:.2f} MB")


if __name__ == "__main__":
    main()
