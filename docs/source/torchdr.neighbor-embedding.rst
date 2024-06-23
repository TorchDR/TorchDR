.. _neighbor-embedding:

.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:


Neighbor Embedding
==================

``TorchDR`` aims to implement most popular neighbor embedding (NE) algorithms.
In this section we briefly go through the main NE algorithms and their variants.


Overview of NE via Attraction and Repulsion
-------------------------------------------

NE objectives share a common structure: they aim to minimize the sum of an attractive term :math:`\mathcal{L}_{\mathrm{att}}` and a repulsive term :math:`\mathcal{L}_{\mathrm{rep}}` which often does not depend on the input affinity :math:`\mathbf{P}`. Thus the NE problem can be formulated as a minimization problem of the form:

.. math::

    \min_{\mathbf{Z}} \: \mathcal{L}_{\mathrm{att}}(\mathbf{P}, \mathbf{Q}) + \mathcal{L}_{\mathrm{rep}}(\mathbf{Q})

.. list-table:: 
   :widths: auto
   :header-rows: 1

   * - **Method**
     - **Attractive term** :math:`\mathcal{L}_{\mathrm{att}}`
     - **Repulsive term** :math:`\mathcal{L}_{\mathrm{rep}}`
     - **Affinity input** :math:`\mathbf{P}`
     - **Affinity output** :math:`\mathbf{Q}`

   * - :class:`SNE <torchdr.neighbor_embedding.SNE>` [1]_
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - :math:`\sum_{i} \log(\sum_j Q_{ij})`
     - :class:`EntropicAffinity <EntropicAffinity>`
     - :class:`GibbsAffinity <GibbsAffinity>`

   * - :class:`TSNE <TSNE>` [2]_
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - :math:`\log(\sum_{ij} Q_{ij})`
     - :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>`
     - :class:`StudentAffinity <torchdr.StudentAffinity>`

   * - :class:`InfoTSNE <torchdr.InfoTSNE>` [15]_
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - :math:`\log(\sum_{(ij) \in B} Q_{ij})`
     - :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>`
     - :class:`StudentAffinity <torchdr.StudentAffinity>`

   * - SNEkhorn [3]_
     - :math:`- \sum_{ij} P_{ij} \log Q_{ij}`
     - :math:`\log(\sum_{(ij) \in B} Q_{ij})`
     - :class:`SymmetricEntropicAffinity <torchdr.SymmetricEntropicAffinity>`
     - :class:`DoublyStochasticEntropic <torchdr.DoublyStochasticEntropic>`

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