.. _overview:


DR estimators
=============


Introduction
------------

This package provides a large collection of **dimensionality reduction** (DR) algorithms. DR focuses on solving problems of the form

.. math::

    \min_{\mathbf{Z}} \: \sum_{ij} L( [\mathbf{A_X}]_{ij}, [\mathbf{A_Z}]_{ij})

where 

  - :math:`\mathbf{A_X}` is the pairwise affinity matrix between input samples :math:`(\mathbf{x}_1, ..., \mathbf{x}_n)`.
  - :math:`\mathbf{A_Z}` is the pairwise affinity matrix between low-dimensional embeddings :math:`(\mathbf{z}_1, ..., \mathbf{z}_n)`.
  - :math:`L` is a loss function.


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


Metric MDS
----------

.. math::

    \min_{\mathbf{Z}} \: \sum_{ij} ( [\mathbf{A_X}]_{ij} - [\mathbf{A_Z}]_{ij} )^{2}


Neighbor Embedding
------------------

.. math::

    \min_{\mathbf{Z}} \: - \sum_{ij} [\mathbf{A_X}]_{ij} \log [\mathbf{A_Z}]_{ij}