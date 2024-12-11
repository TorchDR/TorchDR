Torch Dimensionality Reduction
==============================

.. image:: https://github.com/torchdr/torchdr/raw/main/docs/source/figures/torchdr_logo.png
   :width: 800px
   :alt: torchdr logo
   :align: center

|Documentation| |Version| |License| |Python 3.10+| |Pytorch| |Black| |Test Status| |CircleCI| |codecov| 

TorchDR is an open-source **dimensionality reduction (DR)** library using PyTorch. Its goal is to accelerate the development of new DR methods by providing a common simplified framework.

DR aims to construct a **low-dimensional representation (or embedding)** of an input dataset that best preserves its **geometry encoded via a pairwise affinity matrix** . To this end, DR methods **optimize the embedding** such that its **associated pairwise affinity matrix matches the input affinity**. TorchDR provides a general framework for solving problems of this form. Defining a DR algorithm solely requires choosing or implementing an *Affinity* object for both input and embedding as well as an objective function.

Benefits of TorchDR include:

.. list-table:: 
   :widths: auto
   :header-rows: 0

   * - **Modularity**
     - All of it is written in **python** in a **highly modular** way, making it easy to create or transform components.
   * - **Speed**
     - Supports **GPU acceleration**, leverages **sparsity** and **batching** strategies with **contrastive learning** techniques.
   * - **Memory efficiency**
     - Relies on **sparsity** and/or ``pykeops`` [C21]_ symbolic tensors to **avoid memory overflows**.
   * - **Compatibility**
     - Implemented methods are fully **compatible** with the ``sklearn`` [P11]_ API and ``torch`` [P19]_ ecosystem.


Getting Started
---------------

TorchDR offers a **user-friendly API similar to scikit-learn** where dimensionality reduction modules can be called with the ``fit_transform`` method. It seamlessly accepts both NumPy arrays and PyTorch tensors as input, ensuring that the output matches the type and backend of the input.

.. code-block:: python

    from sklearn.datasets import fetch_openml
    from torchdr import PCA, TSNE

    x = fetch_openml("mnist_784").data.astype("float32")

    x_ = PCA(n_components=50).fit_transform(x)
    z = TSNE(perplexity=30).fit_transform(x_)

TorchDR enables **GPU acceleration without memory limitations** thanks to the KeOps library. This can be easily enabled as follows:

.. code-block:: python

    z_gpu = TSNE(perplexity=30, device="cuda", keops=True).fit_transform(x_)

**MNIST example.**
Here is a comparison of various neighbor embedding methods on the MNIST digits dataset.

.. image:: https://github.com/torchdr/torchdr/raw/main/docs/source/figures/mnist_readme.png
   :width: 800px
   :alt: various neighbor embedding methods on MNIST
   :align: center

The code to generate this figure is available `here <https://github.com/TorchDR/TorchDR/tree/main/examples/mnist/panorama_readme.py>`_.

**Single cell example.**
Here is an example of single cell embeddings using TorchDR, where the embeddings are colored by cell type and the number of cells is indicated in each title.

.. image:: https://github.com/torchdr/torchdr/raw/main/docs/source/figures/single_cell_readme.png
   :width: 700px
   :alt: single cell embeddings
   :align: center

The code for this figure is `here <https://github.com/TorchDR/TorchDR/tree/main/examples/single_cell/single_cell_readme.py>`_.


Implemented Features (to date)
------------------------------

Affinities
~~~~~~~~~~

TorchDR features a **wide range of affinities** which can then be used as a building block for DR algorithms. It includes:

* Usual affinities such that scalar product, Gaussian and Student kernels.
* Affinities based on k-NN normalizations such Self-tuning affinities [Z04]_ and MAGIC [V18]_.
* Doubly stochastic affinities with entropic [S67]_ [C13]_ [F19]_ [L21]_ and quadratic [Z23]_ projections.
* Adaptive affinities with entropy control [H02]_ [V13]_ and its symmetric version [V23]_.

Dimensionality Reduction Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Spectral.** TorchDR provides **spectral embeddings** [H04]_ calculated via eigenvalue decomposition of the affinities or their Laplacian.

**Neighbor Embedding.** TorchDR includes various **neighbor embedding methods** such as *SNE* [H02]_, *t-SNE* [V08]_, *t-SNEkhorn* [V23]_, *UMAP* [M18]_ [D21]_, *LargeVis* [T16]_ and *InfoTSNE* [D23]_.

Evaluation Metric
~~~~~~~~~~~~~~~~~~

TorchDR provides efficient GPU-compatible evaluation metrics : *Silhouette score* [R87]_.


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


Citation
--------

If you use TorchDR in your research, please cite the following reference:

.. code-block:: apalike

    Van Assel H., Courty N., Flamary R., Garivier A., Massias M., Vayer T., Vincent-Cuaz C. TorchDR URL: https://torchdr.github.io/

or in Bibtex format :

.. code-block:: bibtex

    @misc{vanassel2024torchdr,
      author = {Van Assel, Hugues and Courty, Nicolas and Flamary, Rémi and Garivier, Aurélien and Massias, Mathurin and Vayer, Titouan and Vincent-Cuaz, Cédric},
      title = {TorchDR},
      url = {https://torchdr.github.io/},
      year = {2024}
    }


References
----------

.. [H02] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [V08] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [V23] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [V13] Max Vladymyrov, Miguel A. Carreira-Perpinan (2013). `Entropic Affinities: Properties and Efficient Numerical Computation <https://proceedings.mlr.press/v28/vladymyrov13.pdf>`_. International Conference on Machine Learning (ICML).

.. [S67] Richard Sinkhorn, Paul Knopp (1967). `Concerning Nonnegative Matrices and Doubly Stochastic Matrices <https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf>`_. Pacific Journal of Mathematics, 21(2), 343-348.

.. [C13] Marco Cuturi (2013). `Sinkhorn Distances: Lightspeed Computation of Optimal Transport <https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf>`_. Advances in Neural Information Processing Systems 26 (NeurIPS).

.. [F19] Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, Gabriel Peyré (2019). `Interpolating between Optimal Transport and MMD using Sinkhorn Divergences <https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf>`_. International Conference on Artificial Intelligence and Statistics (AISTATS).

.. [M18] Leland McInnes, John Healy, James Melville (2018). `UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction <https://arxiv.org/abs/1802.03426>`_. arXiv preprint arXiv:1802.03426.

.. [Z23] Stephen Zhang, Gilles Mordant, Tetsuya Matsumoto, Geoffrey Schiebinger (2023). `Manifold Learning with Sparse Regularised Optimal Transport <https://arxiv.org/abs/2307.09816>`_. arXiv preprint.

.. [H04] Ham, J., Lee, D. D., Mika, S., & Schölkopf, B. (2004). `A Kernel View of the Dimensionality Reduction of Manifolds <https://icml.cc/Conferences/2004/proceedings/papers/296.pdf>`_. In Proceedings of the twenty-first international conference on Machine learning (ICML).

.. [D21] Sebastian Damrich, Fred Hamprecht (2021). `On UMAP's True Loss Function <https://proceedings.neurips.cc/paper/2021/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf>`_. Advances in Neural Information Processing Systems 34 (NeurIPS).

.. [T16] Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). `Visualizing Large-Scale and High-Dimensional Data <https://dl.acm.org/doi/pdf/10.1145/2872427.2883041?casa_token=9ybi1tW9opcAAAAA:yVfVBu47DYa5_cpmJnQZm4PPWaTdVJgRu2pIMqm3nvNrZV5wEsM9pde03fCWixTX0_AlT-E7D3QRZw>`_. In Proceedings of the 25th international conference on world wide web.

.. [D23] Sebastian Damrich, Jan Niklas Böhm, Fred Hamprecht, Dmitry Kobak (2023). `From t-SNE to UMAP with Contrastive Learning <https://openreview.net/pdf?id=B8a1FcY0vi>`_. International Conference on Learning Representations (ICLR).

.. [L21] Landa, B., Coifman, R. R., & Kluger, Y. (2021). `Doubly Stochastic Normalization of the Gaussian Kernel is Robust to Heteroskedastic Noise <https://epubs.siam.org/doi/abs/10.1137/20M1342124?journalCode=sjmdaq>`_. SIAM journal on mathematics of data science, 3(1), 388-413.

.. [C21] Charlier, B., Feydy, J., Glaunes, J. A., Collin, F. D., & Durif, G. (2021). `Kernel Operations on the GPU, with Autodiff, without Memory Overflows <https://www.jmlr.org/papers/volume22/20-275/20-275.pdf>`_. Journal of Machine Learning Research, 22 (JMLR).

.. [P19] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). `Pytorch: An imperative style, high-performance deep learning library <https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf>`_. Advances in neural information processing systems 32 (NeurIPS).

.. [P11] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). `Scikit-learn: Machine learning in Python <https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf?ref=https:/>`_. Journal of machine Learning research, 12 (JMLR).

.. [Z04] Max Zelnik-Manor, L., & Perona, P. (2004). `Self-Tuning Spectral Clustering <https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf>`_. Advances in Neural Information Processing Systems 17 (NeurIPS).

.. [V18] Van Dijk, D., Sharma, R., Nainys, J., Yim, K., Kathail, P., Carr, A. J., ... & Pe’er, D. (2018). `Recovering Gene Interactions from Single-Cell Data Using Data Diffusion <https://www.cell.com/action/showPdf?pii=S0092-8674%2818%2930724-4>`_. Cell, 174(3).

.. [R87] Rousseeuw, P. J. (1987). `Silhouettes: a graphical aid to the interpretation and validation of cluster analysis <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_. Journal of computational and applied mathematics, 20, 53-65.

.. |Documentation| image:: https://img.shields.io/badge/Documentation-blue.svg
   :target: https://torchdr.github.io/
.. |Pytorch| image:: https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white
   :target: https://pytorch.org/get-started/locally/
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/release/python-3100/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Test Status| image:: https://github.com/torchdr/torchdr/actions/workflows/testing.yml/badge.svg
.. |CircleCI| image:: https://dl.circleci.com/status-badge/img/gh/TorchDR/TorchDR/tree/main.svg?style=svg
   :target: https://dl.circleci.com/status-badge/redirect/gh/TorchDR/TorchDR/tree/main
.. |codecov| image:: https://codecov.io/gh/torchdr/torchdr/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/torchdr/torchdr
.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |Version| image:: https://img.shields.io/badge/version-0.1-blue.svg
   :target: https://github.com/TorchDR/TorchDR/releases/tag/0.1
