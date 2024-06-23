.. _api_and_modules:

API and Modules
===============


.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:


DR modules
----------

Base DR module :py:mod:`torchdr.base`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   DRModule

Spectral DR :py:mod:`torchdr.spectral`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   PCA


Affinity matcher :py:mod:`torchdr.affinity_matcher`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   AffinityMatcher
   BatchedAffinityMatcher


Neighbor embedding :py:mod:`torchdr.neighbor_embedding`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SNE
   TSNE
   InfoTSNE


Affinity modules
----------------

Affinities :py:mod:`torchdr.affinities`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple affinities
"""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   GibbsAffinity
   StudentAffinity
   ScalarProductAffinity

Entropic affinities
"""""""""""""""""""
   
.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   EntropicAffinity
   L2SymmetricEntropicAffinity
   SinkhornAffinity
   SymmetricEntropicAffinity

Quadratic affinities
"""""""""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   DoublyStochasticQuadraticAffinity

UMAP affinities
"""""""""""""""

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst
   
   UMAPAffinityIn
   UMAPAffinityOut   

Utils
-----

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
