.. _neighbor-embedding:


Neighbor embedding algorithms
=============


This package aims to implement most popular neighbor embedding (NE) algorithms.
In this section we briefly go through the main NE algorithms and their variants.

We refer to `"A Probabilistic Graph Coupling View of
Dimension Reduction" <https://proceedings.neurips.cc/paper_files/paper/2022/file/45994782a61bb51cad5c2bae36834265-Paper-Conference.pdf>`_
for a probabilistic point of view on these losses.

NE methods can be divided into two main categories: cross-entropy (CE) and binary cross-entropy (BCE) based methods. 


Part I : cross entropy based objectives (t-SNE like)
----------------------------------------------------

.. math::

    \min_{\mathbf{Z}} \: - \sum_{ij} [\mathbf{A_X}]_{ij} \log [\mathbf{A_Z}]_{ij}

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:
   
   torchdr.affinity.EntropicAffinity


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:
   
   torchdr.affinity.StudentAffinity


Part II : binary cross entropy based objectives (UMAP like)
-----------------------------------------------------------



Part III : relating the two families via contrastive learning
-------------------------------------------------------------


`"From t-SNE to UMAP
with contrastive learning" <https://arxiv.org/pdf/2206.01816>`_