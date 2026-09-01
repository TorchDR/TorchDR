# Torch Dimensionality Reduction

<p align="center">
  <img src="https://github.com/torchdr/torchdr/raw/main/docs/source/figures/torchdr_logo.png" width="800" alt="torchdr logo">
</p>

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://torchdr.github.io/)
[![Benchmark](https://img.shields.io/badge/Benchmarks-blue.svg)](https://github.com/TorchDR/TorchDR/tree/main/benchmarks)
[![Version](https://img.shields.io/github/v/release/TorchDR/TorchDR.svg?color=blue&cacheSeconds=3600)](https://github.com/TorchDR/TorchDR/releases)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Test Status](https://github.com/torchdr/torchdr/actions/workflows/testing.yml/badge.svg)]()
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/TorchDR/TorchDR/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/TorchDR/TorchDR/tree/main)
[![codecov](https://codecov.io/gh/torchdr/torchdr/branch/main/graph/badge.svg)](https://codecov.io/gh/torchdr/torchdr)

**TorchDR** is a high-performance dimensionality reduction library built on PyTorch. It provides GPU and multi-GPU accelerated DR methods in a unified framework with a simple, scikit-learn-compatible API.


## Key Features

| Feature | Description |
|---------|-------------|
| **High Performance** | Engineered for speed with GPU acceleration, `torch.compile` support, and optimized algorithms leveraging sparsity and negative sampling. |
| **Multi-GPU Support** | Scale to massive datasets with built-in distributed computing. Use the `torchdr` CLI or `torchrun` for easy multi-GPU execution of compatible modules and methods. |
| **Modular by Design** | Every component is designed to be easily customized, extended, or replaced to fit your specific needs. |
| **Memory-Efficient** | Natively handles sparsity and memory-efficient symbolic operations. Supports PyTorch DataLoader for streaming large datasets. |
| **Seamless Integration** | Fully compatible with the scikit-learn and PyTorch ecosystems. Use familiar APIs and integrate effortlessly into your existing workflows. |
| **Minimal Dependencies** | Requires only PyTorch, NumPy, and scikit‑learn; optionally add Faiss for fast k‑NN or KeOps for symbolic computation. |


## Getting Started

**TorchDR** offers a **user-friendly API similar to scikit-learn** where dimensionality reduction modules can be called with the `fit_transform` method. It seamlessly accepts both NumPy arrays and PyTorch tensors as input, ensuring that the output matches the type and backend of the input.

```python
from sklearn.datasets import fetch_openml
from torchdr import UMAP

x = fetch_openml("mnist_784").data.astype("float32")

z = UMAP(n_neighbors=30).fit_transform(x)
```

**GPU Acceleration**: Set `device="cuda"` to run on GPU. By default (`device="auto"`), TorchDR uses the input data's device.

```python
z = UMAP(n_neighbors=30, device="cuda").fit_transform(x)
```

**Multi-GPU**: Use the `torchdr` CLI to parallelize across GPUs with no code changes:

```bash
torchdr my_script.py            # Use all available GPUs
torchdr --gpus 4 my_script.py   # Use 4 GPUs
```

**torch.compile**: Enable `compile=True` for additional speed on PyTorch 2.0+.

**Backends**: The `backend` parameter controls k-NN and memory-efficient computations:

| Backend | Description |
|---------|-------------|
| `"faiss"` | Fast approximate k-NN via [Faiss](https://github.com/facebookresearch/faiss) **(Recommended)** |
| `"keops"` | Exact symbolic computation via [KeOps](https://www.kernel-operations.io/keops/index.html) with linear memory |
| `None` | Raw PyTorch |

**DataLoader for Large Datasets**: Pass a PyTorch `DataLoader` instead of a tensor to stream data batch-by-batch. **Requires `backend="faiss"`**.

```python
from torch.utils.data import DataLoader, TensorDataset

dataloader = DataLoader(TensorDataset(X), batch_size=10000, shuffle=False)
z = UMAP(backend="faiss").fit_transform(dataloader)
```

## Methods

### Neighbor Embedding

**TorchDR** provides a suite of neighbor embedding methods, optimal for data visualization.

| Method | Complexity | Multi-GPU | Paper |
|--------|:----------:|:---------:|:-----:|
| [`UMAP`](https://torchdr.github.io/dev/gen_modules/torchdr.UMAP.html) | O(n) | ✅ | [↗](https://arxiv.org/abs/1802.03426) |
| [`LargeVis`](https://torchdr.github.io/dev/gen_modules/torchdr.LargeVis.html) | O(n) | ✅ | [↗](https://arxiv.org/abs/1602.00370) |
| [`InfoTSNE`](https://torchdr.github.io/dev/gen_modules/torchdr.InfoTSNE.html) | O(n) | ✅ | [↗](https://arxiv.org/abs/2206.01816) |
| [`PACMAP`](https://torchdr.github.io/dev/gen_modules/torchdr.PACMAP.html) | O(n) | ❌ | [↗](https://arxiv.org/abs/2012.04456) |
| [`SNE`](https://torchdr.github.io/dev/gen_modules/torchdr.SNE.html) | O(n²) | ✅ | [↗](https://papers.nips.cc/paper/2002/hash/6150ccc6069bea6b5716254057a194ef-Abstract.html) |
| [`TSNE`](https://torchdr.github.io/dev/gen_modules/torchdr.TSNE.html) | O(n²) | ✅ | [↗](https://jmlr.org/papers/v9/vandermaaten08a.html) |
| [`TSNEkhorn`](https://torchdr.github.io/dev/gen_modules/torchdr.TSNEkhorn.html) | O(n²) | ❌ | [↗](https://arxiv.org/abs/2305.13797) |
| [`COSNE`](https://torchdr.github.io/dev/gen_modules/torchdr.COSNE.html) | O(n²) | ✅ | [↗](https://arxiv.org/abs/2111.15037) |

> *Note:* Quadratic methods support `backend="keops"` for exact computation with linear memory usage.


### Spectral Embedding

**TorchDR** provides various **spectral embedding** methods: [`PCA`](https://torchdr.github.io/dev/gen_modules/torchdr.PCA.html), [`IncrementalPCA`](https://torchdr.github.io/dev/gen_modules/torchdr.IncrementalPCA.html), [`ExactIncrementalPCA`](https://torchdr.github.io/dev/gen_modules/torchdr.ExactIncrementalPCA.html), [`KernelPCA`](https://torchdr.github.io/dev/gen_modules/torchdr.KernelPCA.html), [`PHATE`](https://torchdr.github.io/dev/gen_modules/torchdr.PHATE.html). `PCA` and `ExactIncrementalPCA` support multi-GPU distributed training via the `distributed="auto"` parameter.


## Benchmarks

Relying on **TorchDR** enables an **orders-of-magnitude improvement in runtime performance** compared to CPU-based implementations. [See the code](https://github.com/TorchDR/TorchDR/blob/main/benchmarks/benchmark_umap_single_cell.py).

<p align="center">
  <img src="https://github.com/torchdr/torchdr/raw/main/docs/source/figures/umap_benchmark_single_cell.png" width="1024" alt="UMAP benchmark on single cell data">
</p>


## Examples

See the [examples](https://github.com/TorchDR/TorchDR/tree/main/examples/) folder for all examples.


**MNIST.** ([Code](https://github.com/TorchDR/TorchDR/tree/main/examples/images/panorama_readme.py))
A comparison of various neighbor embedding methods on the MNIST digits dataset.

<p align="center">
  <img src="https://github.com/torchdr/torchdr/raw/main/docs/source/figures/mnist_readme.png" width="800" alt="various neighbor embedding methods on MNIST">
</p>


<!-- **Single-cell genomics.** ([Code](https://github.com/TorchDR/TorchDR/tree/main/examples/single_cell/single_cell_readme.py))
Visualizing cells using `LargeVis` from `TorchDR`.

<p align="center">
  <img src="https://github.com/torchdr/torchdr/raw/main/docs/source/figures/single_cell.gif" width="700" alt="single cell embeddings">
</p> -->


**CIFAR100.** ([Code](https://github.com/TorchDR/TorchDR/tree/main/examples/images/cifar100.py))
Visualizing the CIFAR100 dataset using DINO features and `TSNE`.

<p align="center">
  <img src="https://github.com/torchdr/torchdr/raw/main/docs/source/figures/cifar100_tsne.png" width="1024" alt="TSNE on CIFAR100 DINO features">
</p>


## Advanced Features

### Affinities

**TorchDR** features a **wide range of affinities** which can then be used as a building block for DR algorithms. It includes:

- Affinities based on k-NN normalizations: [`SelfTuningAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.SelfTuningAffinity.html), [`MAGICAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.MAGICAffinity.html), [`UMAPAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.UMAPAffinity.html), [`PHATEAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.PHATEAffinity.html), [`PACMAPAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.PACMAPAffinity.html).
- Doubly stochastic affinities: [`SinkhornAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.SinkhornAffinity.html), [`DoublyStochasticQuadraticAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.DoublyStochasticQuadraticAffinity.html).
- Adaptive affinities with entropy control: [`EntropicAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.EntropicAffinity.html), [`SymmetricEntropicAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.SymmetricEntropicAffinity.html).


### Evaluation Metrics

**TorchDR** provides efficient GPU-compatible evaluation metrics: [`silhouette_score`](https://torchdr.github.io/dev/gen_modules/torchdr.silhouette_score.html), [`knn_label_accuracy`](https://torchdr.github.io/dev/gen_modules/torchdr.knn_label_accuracy.html), [`neighborhood_preservation`](https://torchdr.github.io/dev/gen_modules/torchdr.neighborhood_preservation.html), [`kmeans_ari`](https://torchdr.github.io/dev/gen_modules/torchdr.kmeans_ari.html).



## Installation

Install the core `torchdr` library from PyPI:

```bash
pip install torchdr  # or: uv pip install torchdr
```

**Note:** `torchdr` does not install `faiss-gpu` or `pykeops` by default. You need to install them separately to use the corresponding backends.

*   **Faiss (Recommended)**: For the fastest k-NN computations, install [Faiss](https://github.com/facebookresearch/faiss). Please follow their [official installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md). A common method is using `conda`:
    ```bash
    conda install -c pytorch -c nvidia faiss-gpu
    ```

*   **KeOps**: For memory-efficient symbolic computations, install [PyKeOps](https://www.kernel-operations.io/keops/index.html).
    ```bash
    pip install pykeops
    ```

### Installation from Source

If you want to use the latest, unreleased version of `torchdr`, you can install it directly from GitHub:

```bash
pip install git+https://github.com/torchdr/torchdr
```


## Finding Help

If you have any questions or suggestions, feel free to open an issue on the [issue tracker](https://github.com/torchdr/torchdr/issues) or contact [Hugues Van Assel](https://huguesva.github.io/) directly.
