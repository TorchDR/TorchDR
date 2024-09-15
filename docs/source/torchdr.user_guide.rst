.. _user_guide:

.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:

User Guide
==========

.. contents:: Table of Contents
   :depth: 1
   :local:


Overview
--------

General Formulation of Dimensionality Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DR aims to construct a low-dimensional representation (or embedding) :math:`\mathbf{Z} = (\mathbf{z}_1, ..., \mathbf{z}_n)^\top` of an input dataset :math:`\mathbf{X} = (\mathbf{x}_1, ..., \mathbf{x}_n)^\top` that best preserves its geometry, encoded via a pairwise affinity matrix :math:`\mathbf{A_X}`. To this end, DR methods optimize :math:`\mathbf{Z}` such that a pairwise affinity matrix in the embedding space (denoted :math:`\mathbf{A_Z}`) matches :math:`\mathbf{A_X}`. This general problem is as follows

.. math::

  \min_{\mathbf{Z}} \: \mathcal{L}( \mathbf{A_X}, \mathbf{A_Z}) \quad \text{(DR)}

where :math:`\mathcal{L}` is typically the :math:`\ell_2` or cross-entropy loss.
Each DR method is thus characterized by a triplet :math:`(\mathcal{L}, \mathbf{A_X}, \mathbf{A_Z})`.

TorchDR is structured around the above formulation :math:`\text{(DR)}`.
Defining a DR algorithm solely requires providing an :class:`Affinity <Affinity>` object for both input and embedding as well as a loss function :math:`\mathcal{L}`.

All modules follow the ``sklearn`` [21]_ API and can be used in `sklearn pipelines <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.


Torch GPU support and automatic differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TorchDR is built on top of ``torch`` [20]_, offering GPU support and automatic differentiation. This foundation enables efficient computations and straightforward implementation of new DR methods.

To utilize GPU support, set :attr:`device="cuda"` when initializing any module. For CPU computations, set :attr:`device="cpu"`.

.. note::

    DR particularly benefits from GPU acceleration as most computations, including affinity calculations and the DR objective, involve matrix reductions that are highly parallelizable.
    

Avoiding memory overflows with KeOps symbolic (lazy) tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Affinities incur a quadratic memory cost, which can be particularly problematic when dealing with large numbers of samples, especially when using GPUs.

To prevent memory overflows, TorchDR relies on ``pykeops`` [19]_ lazy tensors. These tensors are expressed as mathematical formulas, evaluated directly on the data samples. This symbolic representation allows computations to be performed without storing the entire matrix in memory, thereby effectively eliminating any memory limitation.

.. image:: figures/symbolic_matrix.svg
   :width: 800
   :align: center

The above figure is taken from `here <https://github.com/getkeops/keops/blob/main/doc/_static/symbolic_matrix.svg>`_.

Set :attr:`keops=True` as input to any module to use symbolic tensors. For small datasets, setting :attr:`keops=False` allows the computation of the full affinity matrix directly in memory.


Affinities
----------

Affinities are the essential building blocks of dimensionality reduction methods.
TorchDR provides a wide range of affinities, including basic ones such as :class:`GaussianAffinity <torchdr.GaussianAffinity>`, :class:`StudentAffinity <torchdr.StudentAffinity>` and :class:`ScalarProductAffinity <torchdr.ScalarProductAffinity>`.

Base structure
^^^^^^^^^^^^^^

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


Affinities are objects that can directly be called. The outputed affinity matrix is a **square matrix of size (n, n)** where n is the number of input samples.

Here is an example with the :class:`GaussianAffinity <torchdr.GaussianAffinity>`:

    >>> import torch, torchdr
    >>>
    >>> n = 100
    >>> data = torch.randn(n, 2)
    >>> affinity = torchdr.GaussianAffinity()
    >>> affinity_matrix = affinity(data)
    >>> print(affinity_matrix.shape)
    (100, 100)


Spotlight on affinities based on entropic projections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A widely used family of affinities focuses on **controlling the entropy** of the affinity matrix. It is notably a crucial component of Neighbor-Embedding methods (see :ref:`neighbor-embedding-section`).

These affinities are normalized such that each row sums to one, allowing the affinity matrix to be viewed as a **Markov transition matrix**. An **adaptive bandwidth** parameter then determines how the mass from each point spreads to its neighbors. The bandwidth is based on the :attr:`perplexity` hyperparameter which controls the **number of effective neighbors** for each point.

The resulting affinities can be viewed as a **soft approximation of a k-nearest neighbor graph**, where :attr:`perplexity` takes the role of k. This allows for capturing more nuances than binary weights, as closer neighbors receive a higher weight compared to those farther away. Ultimately, :attr:`perplexity` is an interpretable hyperparameter that governs the scale of dependencies represented in the affinity.

The following table outlines the aspects controlled by different formulations of entropic affinities. **Marginal** indicates whether each row of the affinity matrix has a controlled sum. **Symmetry** indicates whether the affinity matrix is symmetric. **Entropy** indicates whether each row of the affinity matrix has controlled entropy, dictated by the :attr:`perplexity` hyperparameter.


.. list-table:: 
   :widths: auto
   :header-rows: 1

   * - **Affinity (associated DR method)**
     - **Marginal**
     - **Symmetry**
     - **Entropy**
   * - :class:`NormalizedGaussianAffinity <NormalizedGaussianAffinity>`
     - ✅
     - ✅
     - ❌
   * - :class:`SinkhornAffinity <SinkhornAffinity>` [5]_ [9]_
     - ✅
     - ✅
     - ❌
   * - :class:`EntropicAffinity <EntropicAffinity>` [1]_
     - ✅
     - ❌
     - ✅
   * - :class:`SymmetricEntropicAffinity <SymmetricEntropicAffinity>` [3]_
     - ✅
     - ✅
     - ✅

More details on these affinities can be found in the `SNEkhorn paper <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_.


.. minigallery:: torchdr.EntropicAffinity
    :add-heading: Examples using ``EntropicAffinity``:


Other various affinities
^^^^^^^^^^^^^^^^^^^^^^^^

TorchDR features other affinities that can be used in various contexts.

For instance, the UMAP [8]_ algorithm relies on the affinities :class:`UMAPAffinityIn <torchdr.UMAPAffinityIn>` for the input data and :class:`UMAPAffinityOut <torchdr.UMAPAffinityOut>` in the embedding space. :class:`UMAPAffinityIn <torchdr.UMAPAffinityIn>` follows a similar construction as entropic affinities to ensure a constant number of effective neighbors, with :attr:`n_neighbors` playing the role of the :attr:`perplexity` hyperparameter.

Another example is the doubly stochastic normalization of a base affinity under the :math:`\ell_2` geometry that has recently been proposed for DR [10]_. This method is analogous to :class:`SinkhornAffinity <torchdr.SinkhornAffinity>` where the Shannon entropy is replaced by the :math:`\ell_2` norm to recover a sparse affinity.
It is available at :class:`DoublyStochasticQuadraticAffinity <torchdr.DoublyStochasticQuadraticAffinity>`.


Dimensionality Reduction Modules
--------------------------------

TorchDR provides a wide range of dimensionality reduction (DR) methods. All DR estimators inherit the structure of the :meth:`DRModule` class:


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.DRModule

They are :class:`sklearn.base.BaseEstimator` and :class:`sklearn.base.TransformerMixin` classes which can be called with the ``fit_transform`` method.

Outside of :ref:`spectral-section`, a closed-form solution to the DR problem is typically not available. The problem can then be solved using `gradient-based optimizers <https://pytorch.org/docs/stable/optim.html>`_.

The following classes serve as parent classes for this approach, requiring the user to provide affinity objects for the input and output spaces, referred to as :attr:`affinity_in` and :attr:`affinity_out`.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.AffinityMatcher


In what follows we briefly present two families of DR algorithms: neighbor embedding methods and spectral methods.

.. _spectral-section:


Spectral methods
^^^^^^^^^^^^^^^^

Spectral methods correspond to choosing the scalar product affinity :math:`[\mathbf{A_X}]_{ij} = \langle \mathbf{z}_i, \mathbf{z}_j \rangle` for the embeddings and the :math:`\ell_2` loss.

.. math::

    \min_{\mathbf{Z}} \: \sum_{ij} ( [\mathbf{A_X}]_{ij} - \langle \mathbf{z}_i, \mathbf{z}_j \rangle )^{2}

When :math:`\mathbf{A_X}` is positive semi-definite, this problem is commonly known as kernel Principal Component Analysis [11]_ and an optimal solution is given by 

.. math::

    \mathbf{Z}^{\star} = (\sqrt{\lambda_1} \mathbf{v}_1, ..., \sqrt{\lambda_d} \mathbf{v}_d)^\top

where :math:`\lambda_1, ..., \lambda_d` are the largest eigenvalues of the centered kernel matrix :math:`\mathbf{A_X}` and :math:`\mathbf{v}_1, ..., \mathbf{v}_d` are the corresponding eigenvectors.

.. note::

    PCA (available at :class:`torchdr.PCA`) corresponds to choosing :math:`[\mathbf{A_X}]_{ij} = \langle \mathbf{x}_i, \mathbf{x}_j \rangle`.


.. _neighbor-embedding-section:


Neighbor Embedding
^^^^^^^^^^^^^^^^^^^

TorchDR aims to implement most popular **neighbor embedding (NE)** algorithms.
In these methods, :math:`\mathbf{A_X}` and :math:`\mathbf{A_Z}` can be viewed as **soft neighborhood graphs**, hence the term *neighbor embedding*. 

NE objectives share a common structure: they aim to **minimize** the **weighted sum** of an **attractive term** and a **repulsive term**. Interestingly, the **attractive term** is often the **cross-entropy** between the input and output affinities. Additionally, the **repulsive term** is typically a **function of the output affinities only**. Thus, the NE problem can be formulated as the following minimization problem:

.. math::

    \min_{\mathbf{Z}} \: - \sum_{ij} [\mathbf{A_X}]_{ij} \log [\mathbf{A_Z}]_{ij} + \gamma \mathcal{L}_{\mathrm{rep}}(\mathbf{A_Z}) \:.

In the above, :math:`\mathcal{L}_{\mathrm{rep}}(\mathbf{A_Z})` represents the repulsive part of the loss function while :math:`\gamma` is a hyperparameter that controls the balance between attraction and repulsion. The latter is called :attr:`coeff_repulsion` in TorchDR.

Many NE methods can be represented within this framework. The following table summarizes the ones implemented in TorchDR, detailing their respective repulsive loss function, as well as their input and output affinities.

.. list-table:: 
   :widths: auto
   :header-rows: 1

   * - **Method**
     - **Repulsive term** :math:`\mathcal{L}_{\mathrm{rep}}`
     - **Affinity input** :math:`\mathbf{A_X}`
     - **Affinity output** :math:`\mathbf{A_Z}`

   * - :class:`SNE <SNE>` [1]_
     - :math:`\sum_{i} \log(\sum_j [\mathbf{A_Z}]_{ij})`
     - :class:`EntropicAffinity <EntropicAffinity>`
     - :class:`GaussianAffinity <GaussianAffinity>`

   * - :class:`TSNE <TSNE>` [2]_
     - :math:`\log(\sum_{ij} [\mathbf{A_Z}]_{ij})`
     - :class:`EntropicAffinity <EntropicAffinity>`
     - :class:`StudentAffinity <StudentAffinity>`

   * - :class:`TSNEkhorn <TSNEkhorn>` [3]_
     - :math:`\sum_{ij} [\mathbf{A_Z}]_{ij}`
     - :class:`SymmetricEntropicAffinity <SymmetricEntropicAffinity>`
     - :class:`SinkhornAffinity(base_kernel="student") <SinkhornAffinity>`

   * - :class:`InfoTSNE <InfoTSNE>` [15]_
     - :math:`\sum_i \log(\sum_{j \in N(i)} [\mathbf{A_Z}]_{ij})`
     - :class:`EntropicAffinity <EntropicAffinity>`
     - :class:`StudentAffinity <StudentAffinity>`

   * - :class:`UMAP <UMAP>` [8]_
     - :math:`- \sum_{i, j \in N(i)} \log (1 - [\mathbf{A_Z}]_{ij})`
     - :class:`UMAPAffinityIn <UMAPAffinityIn>`
     - :class:`UMAPAffinityOut <UMAPAffinityOut>`

   * - :class:`LargeVis <LargeVis>` [13]_
     - :math:`- \sum_{i, j \in N(i)} \log (1 - [\mathbf{A_Z}]_{ij})`
     - :class:`EntropicAffinity <EntropicAffinity>`
     - :class:`StudentAffinity <StudentAffinity>`

In the above table, :math:`N(i)` denotes the set of negative samples 
for point :math:`i`. They are usually sampled uniformly at random from the dataset.



References
----------

.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [5] Richard Sinkhorn, Paul Knopp (1967). `Concerning nonnegative matrices and doubly stochastic matrices <https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf>`_. Pacific Journal of Mathematics, 21(2), 343-348.

.. [8] Leland McInnes, John Healy, James Melville (2018). `UMAP: Uniform manifold approximation and projection for dimension reduction <https://arxiv.org/abs/1802.03426>`_. arXiv preprint arXiv:1802.03426.

.. [9] Yao Lu, Jukka Corander, Zhirong Yang (2019). `Doubly Stochastic Neighbor Embedding on Spheres <https://www.sciencedirect.com/science/article/pii/S0167865518305099>`_. Pattern Recognition Letters 128 : 100-106.

.. [10] Stephen Zhang, Gilles Mordant, Tetsuya Matsumoto, Geoffrey Schiebinger (2023). `Manifold Learning with Sparse Regularised Optimal Transport <https://arxiv.org/abs/2307.09816>`_. arXiv preprint.

.. [11] Ham, J., Lee, D. D., Mika, S., & Schölkopf, B. (2004). `A kernel view of the dimensionality reduction of manifolds <https://icml.cc/Conferences/2004/proceedings/papers/296.pdf>`_. In Proceedings of the twenty-first international conference on Machine learning (ICML).

.. [13] Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). `Visualizing Large-Scale and High-Dimensional Data <https://dl.acm.org/doi/pdf/10.1145/2872427.2883041?casa_token=9ybi1tW9opcAAAAA:yVfVBu47DYa5_cpmJnQZm4PPWaTdVJgRu2pIMqm3nvNrZV5wEsM9pde03fCWixTX0_AlT-E7D3QRZw>`_. In Proceedings of the 25th international conference on world wide web.

.. [15] Sebastian Damrich, Jan Niklas Böhm, Fred Hamprecht, Dmitry Kobak (2023). `From t-SNE to UMAP with contrastive learning <https://openreview.net/pdf?id=B8a1FcY0vi>`_. International Conference on Learning Representations (ICLR).

.. [19] Charlier, B., Feydy, J., Glaunes, J. A., Collin, F. D., & Durif, G. (2021). `Kernel Operations on the GPU, with Autodiff, without Memory Overflows <https://www.jmlr.org/papers/volume22/20-275/20-275.pdf>`_. Journal of Machine Learning Research (JMLR).

.. [20] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). `Pytorch: An imperative style, high-performance deep learning library <https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf>`_. Advances in neural information processing systems 32 (NeurIPS).

.. [21] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). `Scikit-learn: Machine learning in Python <https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf?ref=https:/>`_. Journal of machine Learning research, 12 (JMLR).