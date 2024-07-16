.. _api_and_modules:

API and Modules
===============


.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:


Dimensionality Reduction ``sklearn`` compatible estimators
----------------------------------------------------------

TorchDR provides a set of classes that are compatible with the ``sklearn`` API. 
Note that the ``torchdr`` classes work seamlessly with both Numpy and PyTorch tensors and
provide additional functionalities that are not available in the ``sklearn``
classes. For example, the ``torchdr`` classes can be used with GPU
acceleration using `device='cuda'` parameter and LazyTensor objects that
allows to perform DR on large datasets that do not fit in memory with
`keops=True`.


Spectral Embedding
^^^^^^^^^^^^^^^^^^

Those classes are used to perform classical spectral embedding of the data. They
are equivalent to use the :class:`torchdr.AffinityMatcher` with a specific
affinity on the data and a :class:`torchdr.ScalarProductAffinity` on the reduced
dimension data.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   PCA


Neighbor Embedding
^^^^^^^^^^^^^^^^^^


Classical Neighbor Embedding Methods
"""""""""""""""""""""""""""""""""""""
   
.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SNE
   TSNE
   TSNEkhorn


Noise Contrastive Neighbor Embedding Methods
"""""""""""""""""""""""""""""""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   InfoTSNE
   LargeVis
   UMAP



Advanced dimensionality reduction with ``torchdr``
--------------------------------------------------

TorchDR provides a set of generic classes that can be used to implement new
dimensionality reduction methods. These classes provide a more flexible way to
implement new methods.


Base classes 
^^^^^^^^^^^^

The :class:`torchdr.DRModule` class is the base class for a dimensionality
reduction estimator. It is the base class for all the DR classes in TorchDR.
:class:`torchdr.AffinityMatcher` is the basis class for all the DR methods that
use generic affinities. Both are compatible with
`sklearn` API.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DRModule
   AffinityMatcher
   

Base Neighbor embedding Modules
"""""""""""""""""""""""""""""""

Neighbor embedding methods are based on the idea of preserving the local
structure of the data. The following classes are the base classes for the
neighbor embedding methods that allows for specific and more efficient
implementations such as sparsity acceleration with
:class:`torchdr.SparseNeighborEmbedding` and stochastic optimization with
:class:`torchdr.SampledNeighborEmbedding`.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NeighborEmbedding
   SparseNeighborEmbedding
   SampledNeighborEmbedding
   


Affinity classes
^^^^^^^^^^^^^^^^

The following classes are used to compute the affinities between the data points
in the high-dimensional and low dimensional spaces. The affinities are used to
define the similarity between the data points and are used to preserve the
structure of the data in the reduced dimension space.


Simple Affinities
"""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   GaussianAffinity
   StudentAffinity
   ScalarProductAffinity
   SelfTuningAffinity

Entropic Affinities
"""""""""""""""""""
   
.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   NormalizedGaussianAffinity
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
the pairwise distances between the data points, binary search, and false
position optimization.

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst
   
   pairwise_distances
   binary_search
   false_position


References
----------

.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [5] Richard Sinkhorn, Paul Knopp (1967). `Concerning nonnegative matrices and doubly stochastic matrices <https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf>`_. Pacific Journal of Mathematics, 21(2), 343-348.

.. [8] Leland McInnes, John Healy, James Melville (2018). `UMAP: Uniform manifold approximation and projection for dimension reduction <https://arxiv.org/abs/1802.03426>`_. arXiv preprint arXiv:1802.03426.

.. [12] Sebastian Damrich, Fred Hamprecht (2021). `On UMAP's True Loss Function <https://proceedings.neurips.cc/paper/2021/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf>`_. Advances in Neural Information Processing Systems 34 (NeurIPS).

.. [13] Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). `Visualizing Large-Scale and High-Dimensional Data <https://dl.acm.org/doi/pdf/10.1145/2872427.2883041?casa_token=9ybi1tW9opcAAAAA:yVfVBu47DYa5_cpmJnQZm4PPWaTdVJgRu2pIMqm3nvNrZV5wEsM9pde03fCWixTX0_AlT-E7D3QRZw>`_. In Proceedings of the 25th international conference on world wide web.

.. [15] Sebastian Damrich, Jan Niklas Böhm, Fred Hamprecht, Dmitry Kobak (2023). `From t-SNE to UMAP with contrastive learning <https://openreview.net/pdf?id=B8a1FcY0vi>`_. International Conference on Learning Representations (ICLR).

.. [22] Max Zelnik-Manor, L., & Perona, P. (2004). `Self-Tuning Spectral Clustering <https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf>`_. Advances in Neural Information Processing Systems 17 (NeurIPS).
