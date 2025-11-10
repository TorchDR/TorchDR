.. _api_and_modules:

API and Modules
===============


.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:


Dimensionality Reduction ``sklearn`` Compatible Estimators
-----------------------------------------------------------

TorchDR provides a set of classes that are compatible with the ``sklearn`` API.
For example, running :class:`TSNE <TSNE>` can be done in the exact same way as running
:class:`sklearn.manifold.TSNE <sklearn.manifold.TSNE>` with the same parameters.
Note that the TorchDR classes work seamlessly with both Numpy and PyTorch tensors.

For all methods, TorchDR provides the ability to use GPU acceleration using
``device='cuda'`` as well as LazyTensor objects that allows to fit large scale models
directly on the GPU memory without overflows using ``keops=True``.

TorchDR supports a variety of dimensionality reduction methods. They are presented in the following sections.


Spectral Embedding
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   PCA
   IncrementalPCA
   ExactIncrementalPCA
   KernelPCA
   PHATE


Neighbor Embedding
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   UMAP
   LargeVis
   PACMAP
   InfoTSNE
   SNE
   TSNE
   TSNEkhorn
   COSNE


Advanced Dimensionality Reduction with TorchDR
-----------------------------------------------

TorchDR provides a set of generic classes that can be used to implement new
dimensionality reduction methods. These classes provide a modular and extensible framework that allows you to focus on the core components of your method.

Base Classes
^^^^^^^^^^^^

The :class:`torchdr.DRModule` class is the base class for a dimensionality
reduction estimator. It is the base class for all the DR classes in TorchDR.

:class:`torchdr.AffinityMatcher` is the base class for all the DR methods that
use gradient-based optimization to minimize a loss function constructed from
two affinities in input and embedding spaces.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DRModule
   AffinityMatcher


Base Neighbor Embedding Modules
"""""""""""""""""""""""""""""""

Neighbor embedding base modules inherit from the :class:`torchdr.AffinityMatcher`
class.
:class:`torchdr.SparseNeighborEmbedding` relies on the sparsity of the
input affinity to compute the attractive term in linear time.
:class:`torchdr.NegativeSamplingNeighborEmbedding` inherits from this class and adds the possibility to
approximate the repulsive term of the loss via negative samples.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NeighborEmbedding
   SparseNeighborEmbedding
   NegativeSamplingNeighborEmbedding


Affinity Classes
^^^^^^^^^^^^^^^^

Simple Affinities
"""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NormalizedGaussianAffinity
   NormalizedStudentAffinity


Affinities Normalized by kNN Distances
"""""""""""""""""""""""""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SelfTuningAffinity
   MAGICAffinity
   PotentialAffinity
   PHATEAffinity
   PACMAPAffinity
   UMAPAffinity


Entropic Affinities
"""""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SinkhornAffinity
   EntropicAffinity
   SymmetricEntropicAffinity


Other Affinities
""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DoublyStochasticQuadraticAffinity


Scores
^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   silhouette_score


Utils
^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   pairwise_distances
   binary_search
   false_position
