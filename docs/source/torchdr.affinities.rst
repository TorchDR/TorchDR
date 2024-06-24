.. _affinities:

.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:

Affinities
=============

Affinities are the essential building blocks of dimensionality reduction methods.
``TorchDR`` provides a wide range of affinities, including basic ones such as :class:`GibbsAffinity <torchdr.GibbsAffinity>`, :class:`StudentAffinity <torchdr.StudentAffinity>` and :class:`ScalarProductAffinity <torchdr.ScalarProductAffinity>`.

Base structure
---------------

Affinities inherit the structure of the following :meth:`Affinity` class.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.Affinity

If computations can be performed in log domain, the :meth:`LogAffinity` class should be used.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.LogAffinity


All affinities have a :meth:`fit` and :meth:`fit_transform` method that can be used to compute the affinity matrix from a given data matrix. The affinity matrix is a square matrix of size :math:`n \times n` where :math:`n` is the number of samples in the data matrix.

.. code-block:: python
  :linenos:

  import torch
  from torchdr import GibbsAffinity

  n = 100
  data = torch.randn(n, 2)
  affinity = GibbsAffinity(keops=False)
  affinity_matrix = affinity.fit_transform(data)


They also have a :meth:`get_batch` method that can be called when the affinity is fitted. This method takes as input the indices of the samples that should be in the same batch. It returns the affinity matrix divided in blocks given by the batch indices. The output is of size :math:`\text{n_batch} \times \text{batch_size} \times \text{batch_size}` where :math:`\text{n_batch}` is the number of blocks and :math:`\text{batch_size}` is the size of each block.

The number of blocks should be a divisor of the number of samples. Here is an example with 5 blocks of size 20 each:

.. code-block:: python
  :linenos:
  :lineno-start: 8

  batch_size = n // 5
  indices = torch.randperm(n).reshape(-1, batch_size)
  batched_affinity_matrix = affinity.get_batch(indices)
  print(batched_affinity_matrix.shape)

Output:

.. code-block:: text

  torch.Size([5, 20, 20])


Avoid memory overflows with symbolic (lazy) tensors
---------------------------------------------------

Affinities result in a square memory cost, which can be problematic when the number of samples is large.

To prevent memory overflows, ``TorchDR`` relies on ``KeOps`` [19]_ lazy tensors. These tensors are represented as mathematical formulas that are evaluated directly on the data samples. This symbolic tensor representation enables computations without storing the entire matrix in memory.

.. image:: figures/symbolic_matrix.svg
   :width: 800
   :align: center

The above figure is taken from `here <https://github.com/getkeops/keops/blob/main/doc/_static/symbolic_matrix.svg>`_.

Importantly, this allows ``TorchDR`` to execute all operations on the GPU without encountering memory overflow issues.


Affinities based on entropic normalization
------------------------------------------


.. list-table:: 
   :widths: auto
   :header-rows: 1

   * - **Affinity (associated DR method)**
     - **Symmetry**
     - **Marginal**
     - **Entropy**
   * - :class:`EntropicAffinity <torchdr.EntropicAffinity>` (:class:`SNE <torchdr.SNE>`) [1]_
     - ❌
     - ✅
     - ✅
   * - :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>` (:class:`TSNE <torchdr.TSNE>`) [2]_
     - ✅
     - ❌
     - ❌
   * - :class:`SinkhornAffinity <torchdr.SinkhornAffinity>` (DOSNES) [5]_ [9]_
     - ✅
     - ✅
     - ❌
   * - :class:`SymmetricEntropicAffinity <torchdr.SymmetricEntropicAffinity>` (SNEkhorn) [3]_
     - ✅
     - ✅
     - ✅

More details can be found in the `SNEkhorn paper <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_.

Other various affinities
-------------------------

``TorchDR`` features other affinities that can be used in various contexts.

For instance, the UMAP algorithm relies on the affinities :class:`UMAPAffinityIn <torchdr.UMAPAffinityIn>` for the input data and :class:`UMAPAffinityOut <torchdr.UMAPAffinityOut>` in the embedding space.

Another example is the doubly stochastic normalization of a base affinity under the :math:`\ell_2` geometry that has recently been proposed for DR [10]_. It is available at :class:`DoublyStochasticQuadraticAffinity <torchdr.DoublyStochasticQuadraticAffinity>`.


References
----------

.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [5] Richard Sinkhorn, Paul Knopp (1967). `Concerning nonnegative matrices and doubly stochastic matrices <https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf>`_. Pacific Journal of Mathematics, 21(2), 343-348.

.. [9] Yao Lu, Jukka Corander, Zhirong Yang (2019). `Doubly Stochastic Neighbor Embedding on Spheres <https://www.sciencedirect.com/science/article/pii/S0167865518305099>`_. Pattern Recognition Letters 128 : 100-106.

.. [10] Stephen Zhang, Gilles Mordant, Tetsuya Matsumoto, Geoffrey Schiebinger (2023). `Manifold Learning with Sparse Regularised Optimal Transport <https://arxiv.org/abs/2307.09816>`_. arXiv preprint.

.. [19] Charlier, B., Feydy, J., Glaunes, J. A., Collin, F. D., & Durif, G. (2021). `Kernel Operations on the GPU, with Autodiff, without Memory Overflows <https://www.jmlr.org/papers/volume22/20-275/20-275.pdf>`_. Journal of Machine Learning Research (JMLR).
