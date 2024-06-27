.. _overview:


Overview
========


Introduction
------------

DR aims to construct a low-dimensional representation (or embedding) :math:`\mathbf{Z} = (\mathbf{z}_1, ..., \mathbf{z}_n)^\top` of an input dataset :math:`\mathbf{X} = (\mathbf{x}_1, ..., \mathbf{x}_n)^\top` that best preserves its geometry, encoded via a pairwise affinity matrix :math:`\mathbf{A_X}`. To this end, DR methods optimize :math:`\mathbf{Z}` such that a pairwise affinity matrix in the embedding space (denoted :math:`\mathbf{A_Z}`) matches :math:`\mathbf{A_X}`. This general problem is as follows

.. math::

  \min_{\mathbf{Z}} \: \mathcal{L}( \mathbf{A_X}, \mathbf{A_Z}) \quad \text{(DR)}

where :math:`\mathcal{L}` is typically the :math:`\ell_2`, :math:`\mathrm{KL}` or :math:`\mathrm{BCE}` loss.
Each DR method is thus characterized by a triplet :math:`(\mathcal{L}, \mathbf{A_X}, \mathbf{A_Z})`.

``TorchDR`` is structured around the above formulation :math:`\text{(DR)}`.
Defining a DR algorithm solely requires providing an ``Affinity`` object for both input and embedding as well as a loss function :math:`\mathcal{L}`.

All DR estimators inherit the structure of the :meth:`DRModule` class:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.base.DRModule

They are :class:`sklearn.base.BaseEstimator` and :class:`sklearn.base.TransformerMixin` classes which can be called with the ``fit_transform`` method.


Spectral methods
----------------

.. math::

    \min_{\mathbf{Z}} \: \sum_{ij} ( [\mathbf{A_X}]_{ij} - \langle \mathbf{z}_i, \mathbf{z}_j \rangle )^{2}

This problem is commonly known as kernel Principal Component Analysis [11]_ and an optimal solution is given by 

.. math::

    \mathbf{Z}^{\star} = (\sqrt{\lambda_1} \mathbf{v}_1, ..., \sqrt{\lambda_d} \mathbf{v}_d)^\top

where :math:`\lambda_1, ..., \lambda_d` are the largest eigenvalues of the centered kernel matrix :math:`\mathbf{A_X}` and :math:`\mathbf{v}_1, ..., \mathbf{v}_d` are the corresponding eigenvectors.

.. note::

    PCA (available at :class:`torchdr.PCA`) corresponds to choosing :math:`[\mathbf{A_X}]_{ij} = \langle \mathbf{x}_i, \mathbf{x}_j \rangle`.


Affinity matching methods
-------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   torchdr.affinity_matcher.AffinityMatcher
   torchdr.affinity_matcher.BatchedAffinityMatcher


MDS-like methods
~~~~~~~~~~~~~~~~

They relie on the square loss between (squared) distance matrices :math:`\mathbf{D_X}` and :math:`\mathbf{D_Z}`.

.. math::

    \min_{\mathbf{Z}} \: \sum_{ij} ( [\mathbf{D_X}]_{ij} - [\mathbf{D_Z}]_{ij} )^{2}


Neighbor Embedding
~~~~~~~~~~~~~~~~~

.. math::
 
    \min_{\mathbf{Z}} \: - \sum_{ij} P_{ij} \log Q_{ij} + \gamma \mathcal{L}_{\mathrm{rep}}(\mathbf{Q}) \:.

For more details, see the :ref:`section dedicated to neighbor embedding algorithms <neighbor-embedding>`.





References
----------

.. [11] Ham, J., Lee, D. D., Mika, S., & Schölkopf, B. (2004). `A kernel view of the dimensionality reduction of manifolds <https://icml.cc/Conferences/2004/proceedings/papers/296.pdf>`_. In Proceedings of the twenty-first international conference on Machine learning (ICML).

.. [17] Hugues Van Assel, Thibault Espinasse, Julien Chiquet, & Franck Picard (2022). `A Probabilistic Graph Coupling View of Dimension Reduction <https://proceedings.neurips.cc/paper_files/paper/2022/file/45994782a61bb51cad5c2bae36834265-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 35 (NeurIPS).