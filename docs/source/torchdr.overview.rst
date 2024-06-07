.. _overview:


DR estimators
=============


Introduction
------------

DR aims to construct a low-dimensional representation (or embedding) :math:`\mathbf{Z} = (\mathbf{z}_1, ..., \mathbf{z}_n)^\top` of an input dataset :math:`\mathbf{X} = (\mathbf{x}_1, ..., \mathbf{x}_n)^\top` that best preserves its geometry, encoded via a pairwise affinity matrix :math:`\mathbf{A_X}`. To this end, DR methods optimize :math:`\mathbf{Z}` such that a pairwise affinity matrix in the embedding space (denoted :math:`\mathbf{A_Z}`) matches :math:`\mathbf{A_X}`. This general problem is as follows

.. math::

  \min_{\mathbf{Z}} \: \sum_{ij} L( [\mathbf{A_X}]_{ij}, [\mathbf{A_Z}]_{ij}) \quad \text{(DR)}

where :math:`L` is typically the :math:`\ell_2`, :math:`\mathrm{KL}` or :math:`\mathrm{BCE}` loss.
Each DR method is thus characterized by a triplet :math:`(L, \mathbf{A_X}, \mathbf{A_Z})`.

``TorchDR`` is structured around the above formulation :math:`\text{(DR)}`.
Defining a DR algorithm solely requires providing an ``Affinity`` object for both input and embedding as well as a loss function :math:`L`.

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

.. note::

    PCA (available at :class:`torchdr.spectral.PCA`) corresponds to choosing :math:`[\mathbf{A_X}]_{ij} = \langle \mathbf{x}_i, \mathbf{x}_j \rangle`.


Metric MDS
----------

.. math::

    \min_{\mathbf{Z}} \: \sum_{ij} ( [\mathbf{A_X}]_{ij} - [\mathbf{A_Z}]_{ij} )^{2}


Neighbor Embedding
------------------

.. math::

    \min_{\mathbf{Z}} \: - \sum_{ij} [\mathbf{A_X}]_{ij} \log [\mathbf{A_Z}]_{ij}