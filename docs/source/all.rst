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

Those classes are used to perform classical spectral embedding from a 
:class:`torchdr.Affinity` object defined on the input data. 
They give the same output as using :class:`torchdr.AffinityMatcher` with this same
:class:`torchdr.Affinity` in input space and a :class:`torchdr.ScalarProductAffinity` in
the embedding space. However, :class:`torchdr.AffinityMatcher` relies on a 
gradient-based solver while the spectral embedding classes rely on the
eigendecomposition of the affinity matrix.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   PCA
   KernelPCA
   IncrementalPCA


Neighbor Embedding
^^^^^^^^^^^^^^^^^^

TorchDR supports the following neighbor embedding methods.

   
.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SNE
   TSNE
   TSNEkhorn
   InfoTSNE
   LargeVis
   UMAP



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
class and implement specific strategies that are common to all neighbor embedding
methods such as early exaggeration.

In particular, :class:`torchdr.SparseNeighborEmbedding` relies on the sparsity of the 
input affinity to compute the attractive term in linear time. :class:`torchdr.SampledNeighborEmbedding` inherits from this class and adds the possibility to
approximate the repulsive term of the loss via negative samples.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NeighborEmbedding
   SparseNeighborEmbedding
   SampledNeighborEmbedding
   

Affinity Classes
^^^^^^^^^^^^^^^^

The following classes are used to compute the affinities between the data points.
Broadly speaking, they define a notion of similarity between samples.


Simple Affinities
"""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   GaussianAffinity
   StudentAffinity
   ScalarProductAffinity
   NormalizedGaussianAffinity
   NormalizedStudentAffinity
   

Affinities Normalized by kNN Distances
"""""""""""""""""""""""""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SelfTuningAffinity
   MAGICAffinity


Entropic Affinities
"""""""""""""""""""
   
.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   SinkhornAffinity
   EntropicAffinity
   SymmetricEntropicAffinity

Quadratic Affinities
"""""""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   DoublyStochasticQuadraticAffinity

UMAP Affinities
"""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   UMAPAffinityIn
   UMAPAffinityOut   

Utils
^^^^^

The following classes are used to perform various operations such as computing
the pairwise distances between the data points as well as solving root search problems.

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst
   
   pairwise_distances
   binary_search
   false_position
