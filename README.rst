
.. image:: https://github.com/torchdr/torchdr/raw/main/docs/source/figures/torchdr_logo.png
   :width: 800px
   :alt: torchdr logo
   :align: center


|Pytorch| |Black| |Test Status| |codecov| |License| |Python 3.10+|


.. .. contents:: Table of Contents
..    :depth: 1
..    :local:

..   **Introduction**_
..   | **Implemented Methods**_
..   | **Documentation**_
..   | **References**_

.. [**Introduction**](#introduction) | [**Implemented Methods**](#implemented-methods) | [**Documentation**](#documentation) | [**References**](#references)

.. warning::
  This library is currently in a phase of active development. All features are subject to change without prior notice. If you are interested in collaborating, please feel free to reach out by opening an issue or starting a discussion.

``TorchDR`` is an open-source dimensionality reduction (DR) library using PyTorch. Its goal is to accelerate the development of new DR methods by providing a common simplified framework.

.. _Introduction:

Introduction
============


DR aims to construct a low-dimensional representation (or embedding) :math:`\mathbf{Z}` of an input dataset :math:`\mathbf{X}` that best preserves its geometry, encoded via a pairwise affinity matrix :math:`\mathbf{A_X}`. To this end, DR methods optimize :math:`\mathbf{Z}` such that a pairwise affinity matrix in the embedding space (denoted :math:`\mathbf{A_Z}`) matches :math:`\mathbf{A_X}`. This general problem is as follows

.. math::

  \min_{\mathbf{Z}} \: \sum_{ij} L( [\mathbf{A_X}]_{ij}, [\mathbf{A_Z}]_{ij}) \quad \text{(DR)}

where :math:`L` is typically the :math:`\ell_2`, :math:`\mathrm{KL}` or :math:`\mathrm{BCE}` loss.
Each DR method is thus characterized by a triplet :math:`(L, \mathbf{A_X}, \mathbf{A_Z})`.

``TorchDR`` is structured around the above formulation :math:`\text{(DR)}`.
Defining a DR algorithm solely requires providing an ``Affinity`` object for both input and embedding as well as a loss function $`L`$.
Code for other aspects, including optimization, is shared across methods. It ensures a fair benchmarking focusing on core differences.

Advantages of ``TorchDR`` also include:

.. list-table:: 
   :widths: auto
   :header-rows: 0

   * - **Modularity**
     - All of it is written in python in a highly modular way, making it easy to create or transform components.
   * - **Speed**
     - Supports GPU acceleration and batching strategies with contrastive learning techniques.
   * - **Memory efficiency**
     - Relies on `KeOps <https://www.kernel-operations.io/keops/index.html>`_ symbolic tensors to completely avoid memory overflows.
   * - **Compatibility**
     - Implemented methods are fully compatible with the ``scikit-learn`` API and ``torch`` ecosystem.
   * - **Parametric estimators**
     - Neural estimators are seamlessly integrated for all methods.


.. _Implemented Methods:

Implemented Methods
===================


* SNE [1]_
* t-SNE [2]_
* SNEkhorn / t-SNEkhorn [3]_
* UMAP [8]_


.. _Documentation:

Documentation
=============


The ``TorchDR`` documentation can be found [here](https://torchdr.github.io/).


.. _References:

References
==========


.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [4] Max Vladymyrov, Miguel A. Carreira-Perpinan (2013). `Entropic Affinities: Properties and Efficient Numerical Computation <https://proceedings.mlr.press/v28/vladymyrov13.pdf>`_. International Conference on Machine Learning (ICML).

.. [5] Richard Sinkhorn, Paul Knopp (1967). `Concerning nonnegative matrices and doubly stochastic matrices <https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf>`_. Pacific Journal of Mathematics, 21(2), 343-348.

.. [6] Marco Cuturi (2013). `Sinkhorn Distances: Lightspeed Computation of Optimal Transport <https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf>`_. Advances in Neural Information Processing Systems 26 (NeurIPS).

.. [7] Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, Gabriel Peyré (2019). `Interpolating between Optimal Transport and MMD using Sinkhorn Divergences <https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf>`_. International Conference on Artificial Intelligence and Statistics (AISTATS).

.. [8] Leland McInnes, John Healy, James Melville (2018). `UMAP: Uniform manifold approximation and projection for dimension reduction <https://arxiv.org/abs/1802.03426>`_. arXiv preprint arXiv:1802.03426.

.. [9] Yao Lu, Jukka Corander, Zhirong Yang. `Doubly Stochastic Neighbor Embedding on Spheres <https://www.sciencedirect.com/science/article/pii/S0167865518305099>`_. Pattern Recognition Letters 128 (2019): 100-106.

.. [10] Stephen Zhang, Gilles Mordant, Tetsuya Matsumoto, Geoffrey Schiebinger. `Manifold Learning with Sparse Regularised Optimal Transport <https://arxiv.org/abs/2307.09816>`_. arXiv preprint (2023).



.. |Pytorch| image:: https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white
    :target: https://pytorch.org/get-started/locally/
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Test Status| image:: https://github.com/torchdr/torchdr/actions/workflows/testing.yml/badge.svg
.. |codecov| image:: https://codecov.io/gh/torchdr/torchdr/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/torchdr/torchdr
.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause