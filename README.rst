Torch Dimensionality Reduction
==============================

.. image:: https://github.com/torchdr/torchdr/raw/main/docs/source/figures/torchdr_logo.png
   :width: 800px
   :alt: torchdr logo
   :align: center

|Documentation| |Benchmark| |Version| |License| |Python 3.10+| |Pytorch| |Ruff| |Test Status| |CircleCI| |codecov|

``TorchDR`` is an open-source **dimensionality reduction (DR)** library using PyTorch. Its goal is to provide **fast GPU-compatible** implementations of DR algorithms, as well as to accelerate the development of new DR methods by providing a **common simplified framework**.

DR aims to construct a **low-dimensional representation (or embedding)** of an input dataset that best preserves its **geometry encoded via a pairwise affinity matrix** . To this end, DR methods **optimize the embedding** such that its **associated pairwise affinity matrix matches the input affinity**. ``TorchDR`` provides a general framework for solving problems of this form. Defining a DR algorithm solely requires choosing or implementing an *Affinity* object for both input and embedding as well as an objective function.

Benefits of ``TorchDR`` include:

.. list-table::
   :widths: auto
   :header-rows: 0

   * - **Speed**
     - Supports **GPU acceleration**, leverages **sparsity** and **sampling** strategies with **contrastive learning** techniques.
   * - **Modularity**
     - All of it is written in **python** in a **highly modular** way, making it easy to create or transform components.
   * - **Memory efficiency**
     - Relies on **sparsity** and/or **symbolic tensors** to **avoid memory overflows**.
   * - **Compatibility**
     - Implemented methods are fully **compatible** with the ``sklearn`` API and ``torch`` ecosystem.


Getting Started
---------------

``TorchDR`` offers a **user-friendly API similar to scikit-learn** where dimensionality reduction modules can be called with the ``fit_transform`` method. It seamlessly accepts both NumPy arrays and PyTorch tensors as input, ensuring that the output matches the type and backend of the input.

.. code-block:: python

    from sklearn.datasets import fetch_openml
    from torchdr import PCA, TSNE

    x = fetch_openml("mnist_784").data.astype("float32")

    x_ = PCA(n_components=50).fit_transform(x)
    z = TSNE(perplexity=30).fit_transform(x_)


``TorchDR`` is fully **GPU compatible**, allowing a **significant speed up** when a GPU is available. To allow computations on the gpu, simply set ``device="cuda"`` as in the following example:

.. code-block:: python

    z_gpu = TSNE(perplexity=30, device="cuda").fit_transform(x_)



Backends
--------

The ``backend`` keyword specifies which tool to use for handling kNN computations and memory-efficient symbolic computations.

- To perform symbolic tensor computations on the GPU without memory limitations, you can leverage the `KeOps Library <https://www.kernel-operations.io/keops/index.html>`_. This library also allows computing kNN graphs. To enable KeOps, set ``backend="keops"``.
- Alternatively, you can use ``backend="faiss"`` to rely on `Faiss <https://github.com/facebookresearch/faiss>`_ for fast kNN computations.
- Finally, setting ``backend=None`` will use raw PyTorch for all computations.


+------------------+-----------+----------+------------------------+-------------------------+------------------------+-------------------------+
| Dataset          | Samples   | Features | Torchdr UMAP (sec)     | Classic UMAP (sec)      | Torchdr UMAP (MB)      | Classic UMAP (MB)       |
+==================+===========+==========+========================+=========================+========================+=========================+
| Macosko          | 44,808    | 50       | 7.7                    | 61.3                    | 100.4                  | 410.9                   |
+------------------+-----------+----------+------------------------+-------------------------+------------------------+-------------------------+
| 10x Mouse Zheng  | 1,306,127 | 50       | 184.4                  | 1910.4                  | 2699.7                 | 11278.1                 |
+------------------+-----------+----------+------------------------+-------------------------+------------------------+-------------------------+




Examples
--------

See the `examples <https://github.com/TorchDR/TorchDR/tree/main/examples/>`_ folder for all examples.


**MNIST.** (`Code <https://github.com/TorchDR/TorchDR/tree/main/examples/images/panorama_readme.py>`_)
A comparison of various neighbor embedding methods on the MNIST digits dataset.

.. image:: https://github.com/torchdr/torchdr/raw/main/docs/source/figures/mnist_readme.png
   :width: 800px
   :alt: various neighbor embedding methods on MNIST
   :align: center


**Single-cell genomics.** (`Code <https://github.com/TorchDR/TorchDR/tree/main/examples/single_cell/single_cell_readme.py>`_)
Visualizing cells using ``TorchDR``. Embeddings are colored by cell type.

.. image:: https://github.com/torchdr/torchdr/raw/main/docs/source/figures/single_cell_readme.png
   :width: 700px
   :alt: single cell embeddings
   :align: center


**CIFAR100.** (`Code <https://github.com/TorchDR/TorchDR/tree/main/examples/images/cifar100.py>`_)
Visualizing the CIFAR100 dataset using DINO features and TSNE.

.. image:: https://github.com/torchdr/torchdr/raw/main/docs/source/figures/cifar100_tsne.png
   :width: 1024px
   :alt: TSNE on CIFAR100 DINO features
   :align: center



Implemented Features (to date)
------------------------------

Affinities
~~~~~~~~~~

``TorchDR`` features a **wide range of affinities** which can then be used as a building block for DR algorithms. It includes:

* Usual affinities such that scalar product, Gaussian and Student kernels.
* Affinities based on k-NN normalizations such as *Self-Tuning Affinities* and *MAGIC*.
* Doubly stochastic affinities with entropic and quadratic projections.
* Adaptive affinities with entropy control (*Entropic Affinities*) and their symmetric version.

Dimensionality Reduction Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Spectral.** ``TorchDR`` provides **spectral embeddings** calculated via eigenvalue decomposition of the affinities or their Laplacian (PCA, KernelPCA, IncrementalPCA).

**Neighbor Embedding.** ``TorchDR`` includes various **neighbor embedding methods** such as *SNE*, *TSNE*, *TSNEkhorn*, *UMAP*, *LargeVis* and *InfoTSNE*.

Evaluation Metric
~~~~~~~~~~~~~~~~~~

``TorchDR`` provides efficient GPU-compatible evaluation metrics : *Silhouette score*.


Installation
------------

You can install the toolbox through PyPI with:

.. code-block:: bash

    pip install torchdr

To get the latest version, you can install it from the source code as follows:

.. code-block:: bash

    pip install git+https://github.com/torchdr/torchdr


Finding Help
------------

If you have any questions or suggestions, feel free to open an issue on the
`issue tracker <https://github.com/torchdr/torchdr/issues>`_ or contact `Hugues Van Assel <https://huguesva.github.io/>`_ directly.


.. Citation
.. --------

.. If you use ``TorchDR`` in your research, please cite the following reference:

.. .. code-block:: apalike

..     Van Assel H., Courty N., Flamary R., Garivier A., Massias M., Vayer T., Vincent-Cuaz C. TorchDR URL: https://torchdr.github.io/

.. or in Bibtex format :

.. .. code-block:: bibtex

..     @misc{vanassel2024torchdr,
..       author = {Van Assel, Hugues and Courty, Nicolas and Flamary, Rémi and Garivier, Aurélien and Massias, Mathurin and Vayer, Titouan and Vincent-Cuaz, Cédric},
..       title = {TorchDR},
..       url = {https://torchdr.github.io/},
..       year = {2024}
..     }


.. |Documentation| image:: https://img.shields.io/badge/Documentation-blue.svg
   :target: https://torchdr.github.io/
.. |Benchmark| image:: https://img.shields.io/badge/Benchmarks-blue.svg
   :target: https://github.com/TorchDR/TorchDR/tree/main/benchmarks
.. |Pytorch| image:: https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white
   :target: https://pytorch.org/get-started/locally/
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/release/python-3100/
.. |Test Status| image:: https://github.com/torchdr/torchdr/actions/workflows/testing.yml/badge.svg
.. |CircleCI| image:: https://dl.circleci.com/status-badge/img/gh/TorchDR/TorchDR/tree/main.svg?style=svg
   :target: https://dl.circleci.com/status-badge/redirect/gh/TorchDR/TorchDR/tree/main
.. |codecov| image:: https://codecov.io/gh/torchdr/torchdr/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/torchdr/torchdr
.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |Version| image:: https://img.shields.io/github/v/release/TorchDR/TorchDR.svg?color=blue
   :target: https://github.com/TorchDR/TorchDR/releases
.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
