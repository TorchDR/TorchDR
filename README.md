# TorchDR - PyTorch Dimensionality Reduction 

[![PyPI version](https://badge.fury.io/py/torchdr.svg)](https://badge.fury.io/py/torchdr)
[![PyTorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build Status](https://github.com/torchdr/torchdr/actions/workflows/testing.yml/badge.svg)](https://github.com/torchdr/torchdr/actions)
[![Codecov Status](https://codecov.io/gh/torchdr/torchdr/branch/main/graph/badge.svg)](https://codecov.io/gh/torchdr/torchdr)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)


> [!WARNING]
> This library is currently in a phase of active development. All features are subject to change without prior notice. If you are interested in collaborating, please feel free to reach out by opening an issue or starting a discussion.

TorchDR is an open-source dimensionality reduction library using PyTorch.

Website and documentation: [https://torchdr.github.io/](https://torchdr.github.io/)

Source code: [https://github.com/TorchDR/TorchDR](https://github.com/TorchDR/TorchDR)

## Why ``TorchDR``?

``TorchDR`` aims to accelerate the development of new DR methods by providing a common simplified framework.

**Dimensionality Reduction.** Let $`\mathbf{X} = (\mathbf{x}_1, ... , \mathbf{x}_n)^\top \in \mathbb{R}^{n \times p}`$ be an input dataset of $n$ samples of dimensionality $p$. DR aims to construct a low-dimensional representation (or embedding) $`\mathbf{Z} = (\mathbf{z}_1, ... , \mathbf{z}_n)^\top \in \mathbb{R}^{n \times d}`$ with $d < p$ that preserves a prescribed geometry for the input dataset. This geometry is encoded via a pairwise affinity matrix $`\mathbf{A_X}`$. A basic example is the Gaussian kernel $`[\mathbf{A_X}]_{ij} = \exp(- \| \mathbf{x}_i - \mathbf{x}_j \|^2_2)`$.

To this end, most popular DR methods
optimize $`\mathbf{Z}`$ such that a well-chosen pairwise affinity matrix in
the embedding space (denoted $`\mathbf{A_Z}`$) matches $`\mathbf{A_X}`$. This general problem is as follows
```math
\min_{\mathbf{Z}} \: \sum_{ij} L( [\mathbf{A_X}]_{ij}, [\mathbf{A_Z}]_{ij})
```
where L is typically the $`\ell_2`$, $`\mathrm{KL}`$ or $`\mathrm{BCE}`$ loss.

Each DR method is thus characterized by a triplet $`(L, \mathbf{A_X}, \mathbf{A_Z})`$.

There are several reasons to opt for ``TorchDR`` among which:

|  |  |
| ----- | -------------- |
| **Modularity** | All of it is written in python in a highly modular way, making it easy to create or transform components.|
| **Speed** | Supports GPU acceleration and batching strategies inspired from constrastive learning.|
| **Memory efficiency** | Relies on [``KeOps``](https://www.kernel-operations.io/keops/index.html) symbolic tensors to completely avoid memory overflows. |
| **Compatibility** | Implemented methods are fully compatible with the ``scikit-learn`` API and ``torch`` ecosystem. |
|  |  |


## License

The library is distributed under the 3-Clause BSD license.

## References

[1] Geoffrey Hinton, Sam Roweis (2002). [Stochastic Neighbor Embedding](https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf). Advances in neural information processing systems 15 (NeurIPS).

[2] Laurens van der Maaten, Geoffrey Hinton (2008). [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl). The Journal of Machine Learning Research 9.11 (JMLR).

[3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). [SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities](https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf). Advances in Neural Information Processing Systems 36 (NeurIPS).

[4] Max Vladymyrov, Miguel A. Carreira-Perpinan (2013). [Entropic Affinities: Properties and Efficient Numerical Computation](https://proceedings.mlr.press/v28/vladymyrov13.pdf). International Conference on Machine Learning (ICML).

[5] Richard Sinkhorn, Paul Knopp (1967). [Concerning nonnegative matrices and doubly stochastic matrices](https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf). Pacific Journal of Mathematics, 21(2), 343-348.

[6] Marco Cuturi (2013). [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf). Advances in Neural Information Processing Systems 26 (NeurIPS).

[7] Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, Gabriel Peyré (2019). [Interpolating between Optimal Transport and MMD using Sinkhorn Divergences](https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf). International Conference on Artificial Intelligence and Statistics (AISTATS).


<!-- [] Yao Lu, Jukka Corander, Zhirong Yang. ["Doubly Stochastic Neighbor Embedding on Spheres."](https://www.sciencedirect.com/science/article/pii/S0167865518305099) Pattern Recognition Letters 128 (2019): 100-106.

[] Stephen Zhang, Gilles Mordant, Tetsuya Matsumoto, Geoffrey Schiebinger. ["Manifold Learning with Sparse Regularised Optimal Transport."](https://arxiv.org/abs/2307.09816) arXiv preprint (2023). -->
