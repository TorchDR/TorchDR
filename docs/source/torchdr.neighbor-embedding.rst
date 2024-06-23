.. _neighbor-embedding:

.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:


Neighbor Embedding
==================

``TorchDR`` aims to implement most popular neighbor embedding (NE) algorithms.
In this section we briefly go through the main NE algorithms and their variants.


Overview of NE methods
----------------------

NE methods can be divided into two main categories: cross-entropy (CE) and binary cross-entropy (BCE) based methods.

.. math::

    \min_{\mathbf{Z}} \: - \sum_{ij} [\mathbf{P_X}]_{ij} \log [\mathbf{Q_Z}]_{ij}


.. list-table:: 
   :widths: auto
   :header-rows: 1

   * - **Method**
     - **Attractive term**
     - **Repulsive term**
     - **Affinity input** :math:`\mathbf{P}`
     - **Affinity output** :math:`\mathbf{Q}`

   * - :class:`SNE <torchdr.neighbor_embedding.SNE>` [1]_
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - Student
     - :class:`EntropicAffinity <EntropicAffinity>`
     - :class:`GibbsAffinity <GibbsAffinity>`

   * - :class:`TSNE <TSNE>` [2]_
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - Student
     - :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>`
     - :class:`StudentAffinity <torchdr.StudentAffinity>`

   * - :class:`InfoTSNE <torchdr.InfoTSNE>` [15]_
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - Student
     - :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>`
     - :class:`StudentAffinity <torchdr.StudentAffinity>`

   * - UMAP
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - Student
     - :class:`UMAPAffinityIn <torchdr.UMAPAffinityIn>`
     - :class:`UMAPAffinityOut <torchdr.UMAPAffinityOut>`

   * - LargeVis 
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - Student
     - :class:`UMAPAffinityIn <torchdr.L2SymmetricEntropicAffinity>`
     - :class:`UMAPAffinityOut <torchdr.UMAPAffinityOut>`


References
----------

.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [15] Sebastian Damrich, Jan Niklas Böhm, Fred Hamprecht, Dmitry Kobak (2023). `From t-SNE to UMAP with contrastive learning <https://openreview.net/pdf?id=B8a1FcY0vi>`_. International Conference on Learning Representations (ICLR).