.. _api_and_modules:

API and Modules
===============


.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:

This page provides a complete reference of all TorchDR classes and functions.
For conceptual background, see the :ref:`user_guide`.


Dimensionality Reduction Methods
--------------------------------

TorchDR provides ``sklearn``-compatible estimators that work seamlessly with both NumPy arrays and PyTorch tensors. All methods support GPU acceleration via ``device="cuda"`` and can scale to large datasets using ``backend="faiss"`` or ``backend="keops"``.


Neighbor Embedding
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   UMAP
   TSNE
   InfoTSNE
   LargeVis
   SNE
   TSNEkhorn
   COSNE
   PACMAP


Spectral Methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   PCA
   IncrementalPCA
   ExactIncrementalPCA
   KernelPCA
   PHATE


Affinities
----------

Affinities are the building blocks for constructing the input similarity matrix :math:`\mathbf{P}`.
See :ref:`user_guide` for details on how affinities are used in DR methods.


Adaptive Affinities
^^^^^^^^^^^^^^^^^^^

Affinities that adapt bandwidth based on local neighborhood structure.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   EntropicAffinity
   SymmetricEntropicAffinity
   UMAPAffinity
   PACMAPAffinity
   SelfTuningAffinity
   MAGICAffinity
   PHATEAffinity


Other Normalized Affinities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Other normalized affinity kernels.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NormalizedGaussianAffinity
   NormalizedStudentAffinity
   SinkhornAffinity
   DoublyStochasticQuadraticAffinity


Base Classes
------------

These classes provide the foundation for implementing custom DR methods.


Core Base Classes
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DRModule
   AffinityMatcher


Neighbor Embedding Base Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Base classes for neighbor embedding methods. :class:`NeighborEmbedding` leverages input affinity sparsity for efficient attractive term computation. :class:`NegativeSamplingNeighborEmbedding` adds approximate repulsive term computation via negative sampling.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NeighborEmbedding
   NegativeSamplingNeighborEmbedding


Evaluation Metrics
------------------

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   silhouette_score
   silhouette_samples
   knn_label_accuracy
   neighborhood_preservation
   kmeans_ari


Utils
-----

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   pairwise_distances
   binary_search
   false_position
