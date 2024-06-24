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


All affinities have a :meth:`fit` and :meth:`fit_transform` method that can be used to compute the affinity matrix from a given data matrix. The affinity matrix is a **square matrix of size (n, n)** where n is the number of input samples.

Here is an example with the Gibbs affinity:

.. code-block:: python
  :linenos:

  import torch, torchdr

  n = 100
  data = torch.randn(n, 2)
  affinity = torchdr.GibbsAffinity()
  affinity_matrix = affinity.fit_transform(data)


They also have a :meth:`get_batch` method that can be called when the affinity is fitted. This method takes as input the indices of the samples that should be in the same batch. It returns the **affinity matrix divided in blocks** given by the batch indices. The output is of size **(n_batch, batch_size, batch_size)** where n_batch is the number of blocks and batch_size is the number of samples per block.

The number of blocks should be a divisor of the number of samples. Here is an example with 5 blocks of size 20 each:

.. code-block:: python
  :linenos:
  :lineno-start: 7

  batch_size = n // 5
  indices = torch.randperm(n).reshape(-1, batch_size)
  batched_affinity_matrix = affinity.get_batch(indices)
  print(batched_affinity_matrix.shape)

Output:

.. code-block:: text

  (5, 20, 20)


Avoiding memory overflows with symbolic (lazy) tensors
------------------------------------------------------

Affinities incur a quadratic memory cost, which can be particularly problematic when dealing with large numbers of samples, especially when using GPUs.

To prevent memory overflows, ``TorchDR`` relies on ``KeOps`` [19]_ lazy tensors. These tensors are expressed as mathematical formulas, evaluated directly on the data samples. This symbolic representation allows computations to be performed without storing the entire matrix in memory, thereby effectively eliminating any memory limitation.

.. image:: figures/symbolic_matrix.svg
   :width: 800
   :align: center

The above figure is taken from `here <https://github.com/getkeops/keops/blob/main/doc/_static/symbolic_matrix.svg>`_.

.. note::

    All ``TorchDR`` modules have a ``keops`` parameter that can be set to ``True`` to use symbolic tensors. For small datasets, setting this parameter to ``False`` allows the computation of the full affinity matrix directly in memory.



Affinities based on entropic projections
----------------------------------------

A widely used family of affinities focuses on **controlling the entropy** of the affinity matrix, which is a crucial aspect of SNE-related methods [1]_.

The first step is to ensure that each point has a unit mass, allowing the affinity matrix to be viewed as a **Markov transition matrix**. An **adaptive bandwidth** parameter then determines how the mass from each point spreads to its neighbors. The bandwidth is based on the :attr:`perplexity` hyperparameter which controls the **number of effective neighbors** for each point.

The resulting affinities can be seen as a **soft approximation of a k nearest neighbor graph** where the :attr:`perplexity` plays the role of k. It allows capturing more subtleties than binary weights. Ultimately, the :attr:`perplexity` is an interpretable hyperparameter that determines which scale of dependencies is represented in the affinity.

The following table details the aspects controlled by various formulations of entropic affinities. **Marginal** refers to the row-wise control of mass. **Entropy** relates to the row-wise control of entropy dictated by the :attr:`perplexity` hyperparameter.


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

More details on these affinities can be found in the `SNEkhorn paper <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_.


.. note::
    The above table shows that :class:`SymmetricEntropicAffinity <torchdr.SymmetricEntropicAffinity>` is the proper symmetric version of :class:`EntropicAffinity <torchdr.EntropicAffinity>`.
    However :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>` is more efficient to compute and does not require choosing a learning rate. Hence it can be a useful approximation in practice.


Other various affinities
-------------------------

``TorchDR`` features other affinities that can be used in various contexts.

For instance, the UMAP [8]_ algorithm relies on the affinities :class:`UMAPAffinityIn <torchdr.UMAPAffinityIn>` for the input data and :class:`UMAPAffinityOut <torchdr.UMAPAffinityOut>` in the embedding space. :class:`UMAPAffinityIn <torchdr.UMAPAffinityIn>` follows a similar construction as entropic affinities to ensure a constant number of effective neighbors, with :attr:`n_neighbors` playing the role of the :attr:`perplexity` hyperparameter.

Another example is the doubly stochastic normalization of a base affinity under the :math:`\ell_2` geometry that has recently been proposed for DR [10]_. This method is analogous to :class:`SinkhornAffinity <torchdr.SinkhornAffinity>` where the Shannon entropy is replaced by the :math:`\ell_2` norm to recover a sparse affinity.
It is available at :class:`DoublyStochasticQuadraticAffinity <torchdr.DoublyStochasticQuadraticAffinity>`.


References
----------

.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [5] Richard Sinkhorn, Paul Knopp (1967). `Concerning nonnegative matrices and doubly stochastic matrices <https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf>`_. Pacific Journal of Mathematics, 21(2), 343-348.

.. [8] Leland McInnes, John Healy, James Melville (2018). `UMAP: Uniform manifold approximation and projection for dimension reduction <https://arxiv.org/abs/1802.03426>`_. arXiv preprint arXiv:1802.03426.

.. [9] Yao Lu, Jukka Corander, Zhirong Yang (2019). `Doubly Stochastic Neighbor Embedding on Spheres <https://www.sciencedirect.com/science/article/pii/S0167865518305099>`_. Pattern Recognition Letters 128 : 100-106.

.. [10] Stephen Zhang, Gilles Mordant, Tetsuya Matsumoto, Geoffrey Schiebinger (2023). `Manifold Learning with Sparse Regularised Optimal Transport <https://arxiv.org/abs/2307.09816>`_. arXiv preprint.

.. [19] Charlier, B., Feydy, J., Glaunes, J. A., Collin, F. D., & Durif, G. (2021). `Kernel Operations on the GPU, with Autodiff, without Memory Overflows <https://www.jmlr.org/papers/volume22/20-275/20-275.pdf>`_. Journal of Machine Learning Research (JMLR).
