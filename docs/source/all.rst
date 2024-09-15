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


References
----------

.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [8] Leland McInnes, John Healy, James Melville (2018). `UMAP: Uniform manifold approximation and projection for dimension reduction <https://arxiv.org/abs/1802.03426>`_. arXiv preprint arXiv:1802.03426.

.. [12] Sebastian Damrich, Fred Hamprecht (2021). `On UMAP's True Loss Function <https://proceedings.neurips.cc/paper/2021/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf>`_. Advances in Neural Information Processing Systems 34 (NeurIPS).

.. [13] Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). `Visualizing Large-Scale and High-Dimensional Data <https://dl.acm.org/doi/pdf/10.1145/2872427.2883041?casa_token=9ybi1tW9opcAAAAA:yVfVBu47DYa5_cpmJnQZm4PPWaTdVJgRu2pIMqm3nvNrZV5wEsM9pde03fCWixTX0_AlT-E7D3QRZw>`_. In Proceedings of the 25th international conference on world wide web.

.. [15] Sebastian Damrich, Jan Niklas Böhm, Fred Hamprecht, Dmitry Kobak (2023). `From t-SNE to UMAP with contrastive learning <https://openreview.net/pdf?id=B8a1FcY0vi>`_. International Conference on Learning Representations (ICLR).

.. [22] Max Zelnik-Manor, L., & Perona, P. (2004). `Self-Tuning Spectral Clustering <https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf>`_. Advances in Neural Information Processing Systems 17 (NeurIPS).

.. [23] Van Dijk, D., Sharma, R., Nainys, J., Yim, K., Kathail, P., Carr, A. J., ... & Pe’er, D. (2018). `Recovering Gene Interactions from Single-Cell Data Using Data Diffusion <https://www.cell.com/action/showPdf?pii=S0092-8674%2818%2930724-4>`_. Cell, 174(3).
