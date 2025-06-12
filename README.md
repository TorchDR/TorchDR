# Torch Dimensionality Reduction

<p align="center">
  <img src="https://github.com/torchdr/torchdr/raw/main/docs/source/figures/torchdr_logo.png" width="800" alt="torchdr logo">
</p>

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://torchdr.github.io/)
[![Benchmark](https://img.shields.io/badge/Benchmarks-blue.svg)](https://github.com/TorchDR/TorchDR/tree/main/benchmarks)
[![Version](https://img.shields.io/github/v/release/TorchDR/TorchDR.svg?color=blue)](https://github.com/TorchDR/TorchDR/releases)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Test Status](https://github.com/torchdr/torchdr/actions/workflows/testing.yml/badge.svg)]()
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/TorchDR/TorchDR/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/TorchDR/TorchDR/tree/main)
[![codecov](https://codecov.io/gh/torchdr/torchdr/branch/main/graph/badge.svg)](https://codecov.io/gh/torchdr/torchdr)

`TorchDR` is an open-source **dimensionality reduction (DR)** library using PyTorch. It provides **GPU-accelerated** implementations of popular DR algorithms in a single unified framework.

DR aims to construct a **low-dimensional representation (or embedding)** of an input dataset that best preserves its **geometry encoded via a pairwise affinity matrix**. To this end, DR methods **optimize the embedding** such that its **associated pairwise affinity matrix matches the input affinity**. `TorchDR` provides a general framework for solving problems of this form. Defining a DR algorithm solely requires choosing or implementing an *Affinity* object for both input and embedding as well as an objective function.


## Benefits of TorchDR

- **Speed**: supports **GPU acceleration**, leverages **sparsity** and **contrastive learning** / **negative sampling** techniques.
- **Modularity**: all of it is written in **Python** in a **highly modular** way, making it easy to create or transform components.
- **Memory efficiency**: relies on **sparsity** and/or **symbolic tensors** to **avoid memory overflows**.
- **Compatibility**: implemented methods are fully **compatible** with the `sklearn` and `torch` ecosystems.


## Getting Started

`TorchDR` offers a **user-friendly API similar to scikit-learn** where dimensionality reduction modules can be called with the `fit_transform` method. It seamlessly accepts both `NumPy` arrays and `PyTorch` tensors as input, ensuring that the output matches the type and backend of the input.

```python
from sklearn.datasets import fetch_openml
from torchdr import PCA, UMAP

x = fetch_openml("mnist_784").data.astype("float32")

x_ = PCA(n_components=50).fit_transform(x)
z = UMAP(n_neighbors=30).fit_transform(x_)
```

`TorchDR` is fully **GPU compatible**, enabling **significant speed-ups** when a GPU is available. To run computations on the GPU, simply set `device="cuda"` as shown in the example below:

```python
z_gpu = UMAP(n_neighbors=30 device="cuda").fit_transform(x_)
```


## Methods

### Neighbor Embedding

`TorchDR` provides a suite of **neighbor embedding** methods.

**Linear-time (Contrastive Learning).** State-of-the-art speed on large datasets: [`UMAP`](https://torchdr.github.io/dev/gen_modules/torchdr.UMAP.html), [`LargeVis`](https://torchdr.github.io/dev/gen_modules/torchdr.LargeVis.html), [`InfoTSNE`](https://torchdr.github.io/dev/gen_modules/torchdr.InfoTSNE.html).

**Quadratic-time (Exact Repulsion).** Compute the full pairwise repulsion: [`SNE`](https://torchdr.github.io/dev/gen_modules/torchdr.SNE.html), [`TSNE`](https://torchdr.github.io/dev/gen_modules/torchdr.TSNE.html), [`TSNEkhorn`](https://torchdr.github.io/dev/gen_modules/torchdr.TSNEkhorn.html), [`COSNE`](https://torchdr.github.io/dev/gen_modules/torchdr.COSNE.html).

> *Remark.* For quadratic-time algorithms, `TorchDR` provides exact implementations that scale linearly in memory using `backend=keops`.
> For `TSNE` specifically, one can also explore fast approximations, such as `FIt-SNE` implemented in [tsne-cuda](https://github.com/CannyLab/tsne-cuda), which bypass full pairwise repulsion.


### Spectral Embedding

`TorchDR` provides **spectral embeddings** calculated via eigenvalue decomposition:
- [`PCA`](https://torchdr.github.io/dev/gen_modules/torchdr.PCA.html)
- [`IncrementalPCA`](https://torchdr.github.io/dev/gen_modules/torchdr.IncrementalPCA.html) to handle massive datasets
- [`KernelPCA`](https://torchdr.github.io/dev/gen_modules/torchdr.KernelPCA.html) to leverage any `TorchDR` *Affinity* (see the [Affinities](#affinities) subsection below)


## Benchmarks

Relying on `TorchDR` enables an **orders-of-magnitude improvement in runtime performance** compared to CPU-based implementations. [See the code](https://github.com/TorchDR/TorchDR/blob/main/benchmarks/benchmark_umap_single_cell.py).

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

`TorchDR` features a **wide range of affinities** which can then be used as a building block for DR algorithms. It includes:

- Usual affinities: [`ScalarProductAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.ScalarProductAffinity.html), [`GaussianAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.GaussianAffinity.html), [`StudentAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.StudentAffinity.html) [`CauchyAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.CauchyAffinity.html).
- Affinities based on k-NN normalizations: [`SelfTuningAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.SelfTuningAffinity.html), [`MAGICAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.MAGICAffinity.html).
- Doubly stochastic affinities: [`SinkhornAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.SinkhornAffinity.html), [`DoublyStochasticQuadraticAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.DoublyStochasticQuadraticAffinity.html).
- Adaptive affinities with entropy control: [`EntropicAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.EntropicAffinity.html), [`SymmetricEntropicAffinity`](https://torchdr.github.io/dev/gen_modules/torchdr.SymmetricEntropicAffinity.html).

### Evaluation Metric

`TorchDR` provides efficient GPU-compatible evaluation metrics: [`silhouette_score`](https://torchdr.github.io/dev/gen_modules/torchdr.silhouette_score.html).

### Backends

The `backend` keyword specifies which tool to use for handling kNN computations and memory-efficient symbolic computations.

- Set `backend="faiss"` to rely on [Faiss](https://github.com/facebookresearch/faiss) for fast kNN computations.
- To perform exact symbolic tensor computations on the GPU without memory limitations, you can leverage the [KeOps](https://www.kernel-operations.io/keops/index.html) library. This library also allows computing kNN graphs. To enable KeOps, set `backend="keops"`.
- Finally, setting `backend=None` will use raw PyTorch for all computations.


## Installation

You can install the toolbox through PyPI with:

```bash
pip install torchdr
```

To get the latest version, you can install it from the source code as follows:

```bash
pip install git+https://github.com/torchdr/torchdr
```


## Finding Help

If you have any questions or suggestions, feel free to open an issue on the [issue tracker](https://github.com/torchdr/torchdr/issues) or contact [Hugues Van Assel](https://huguesva.github.io/) directly.
