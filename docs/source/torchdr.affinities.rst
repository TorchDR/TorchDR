.. _affinities:

Affinities
=============

Affinities are the essential building blocks of ``TorchDR``. They inherit the structure of the :meth:`Affinity` class.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.affinity.Affinity

If computations can be performed in log domain, the :meth:`LogAffinity` class should be used.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.affinity.LogAffinity

All affinities have a :meth:`fit` and :meth:`fit_transform` method that can be used to compute the affinity matrix from a given data matrix. The affinity matrix is a square matrix of size :math:`n \times n` where :math:`n` is the number of samples in the data matrix.

>>> import torch


They also have a :meth:`get_batch` method that can be called when the affinity is fitted.


Base structure and simple examples
-----------------------------------



.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.GibbsAffinity
   torchdr.affinity.StudentAffinity
   torchdr.affinity.ScalarProductAffinity



Affinities based on entropic normalization
------------------------------------------


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.EntropicAffinity
   torchdr.affinity.SymmetricEntropicAffinity
   torchdr.affinity.L2SymmetricEntropicAffinity
   torchdr.affinity.DoublyStochasticEntropic


.. list-table:: 
   :widths: auto
   :header-rows: 1

   * - **Affinity**
     - **Symmetry**
     - **Marginal**
     - **Entropy**
   * - Entropic (SNE) [1]_
     - ❌
     - ✅
     - ✅
   * - :math:`\ell_2` - Symmetric Entropic (TSNE) [2]_
     - ✅
     - ❌
     - ❌
   * - Doubly Stochastic (DOSNES) [5]_ [9]_
     - ✅
     - ✅
     - ❌
   * - Symmetric Entropic (SNEkhorn) [3]_
     - ✅
     - ✅
     - ✅


Other various affinities
-------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.DoublyStochasticQuadratic


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   
   torchdr.affinity.UMAPAffinityIn
   torchdr.affinity.UMAPAffinityOut


References
----------

.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [5] Richard Sinkhorn, Paul Knopp (1967). `Concerning nonnegative matrices and doubly stochastic matrices <https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf>`_. Pacific Journal of Mathematics, 21(2), 343-348.

.. [9] Yao Lu, Jukka Corander, Zhirong Yang (2019). `Doubly Stochastic Neighbor Embedding on Spheres <https://www.sciencedirect.com/science/article/pii/S0167865518305099>`_. Pattern Recognition Letters 128 : 100-106.
