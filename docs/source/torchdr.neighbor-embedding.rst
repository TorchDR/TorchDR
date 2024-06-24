.. _neighbor-embedding:

.. currentmodule:: torchdr

.. automodule:: torchdr
   :no-members:
   :no-inherited-members:


Neighbor Embedding
==================

``TorchDR`` aims to implement most popular **neighbor embedding (NE)** algorithms.
In this section we briefly go through the main NE algorithms and their variants.

For consistency with the literature, we will denote the input affinity matrix by :math:`\mathbf{P}` and the output affinity matrix by :math:`\mathbf{Q}`. These affinities can be viewed as **soft neighborhood graphs**, hence the term *neighbor embedding*.


Overview of NE via Attraction and Repulsion
-------------------------------------------

NE objectives share a common structure: they aim to minimize the weighted sum of an attractive term and a repulsive term. Interestingly, the attractive term is often the cross-entropy between the input and output affinities. Additionally, the repulsive term is typically a function of the output affinities only. Thus, the NE problem can be formulated as the following minimization problem:

.. math::

    \min_{\mathbf{Z}} \: - \sum_{ij} P_{ij} \log Q_{ij} + \gamma \mathcal{L}_{\mathrm{rep}}(\mathbf{Q}) \:.

In the above, :math:`\mathcal{L}_{\mathrm{rep}}(\mathbf{Q})` represents the repulsive part of the loss function while :math:`\gamma` is a hyperparameter that controls the balance between attraction and repulsion.

Many NE methods can be represented within this framework. The following table summarizes the ones implemented in ``TorchDR``, detailing their respective repulsive loss function, as well as their input and output affinities.

.. list-table:: 
   :widths: auto
   :header-rows: 1

   * - **Method**
     - **Repulsive term** :math:`\mathcal{L}_{\mathrm{rep}}`
     - **Affinity input** :math:`\mathbf{P}`
     - **Affinity output** :math:`\mathbf{Q}`

   * - :class:`SNE <torchdr.neighbor_embedding.SNE>` [1]_
     - :math:`\sum_{i} \log(\sum_j Q_{ij})`
     - :class:`EntropicAffinity <EntropicAffinity>`
     - :class:`GibbsAffinity <GibbsAffinity>`

   * - :class:`TSNE <TSNE>` [2]_
     - :math:`\log(\sum_{ij} Q_{ij})`
     - :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>`
     - :class:`StudentAffinity <torchdr.StudentAffinity>`

   * - :class:`InfoTSNE <torchdr.InfoTSNE>` [15]_
     - :math:`\log(\sum_{(ij) \in B} Q_{ij})`
     - :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>`
     - :class:`StudentAffinity <torchdr.StudentAffinity>`

   * - SNEkhorn [3]_
     - :math:`\log(\sum_{ij} Q_{ij})`
     - :class:`SymmetricEntropicAffinity <torchdr.SymmetricEntropicAffinity>`
     - :class:`SinkhornAffinity(base_kernel="gaussian") <torchdr.SinkhornAffinity>`

   * - TSNEkhorn [3]_
     - :math:`\log(\sum_{ij} Q_{ij})`
     - :class:`SymmetricEntropicAffinity <torchdr.SymmetricEntropicAffinity>`
     - :class:`SinkhornAffinity(base_kernel="student") <torchdr.SinkhornAffinity>`

   * - UMAP [8]_
     - :math:`- \sum_{ij} \log (1 - Q_{ij})`
     - :class:`UMAPAffinityIn <torchdr.UMAPAffinityIn>`
     - :class:`UMAPAffinityOut <torchdr.UMAPAffinityOut>`

   * - LargeVis [13]_
     - :math:`- \sum_{ij} \log (1 - Q_{ij})`
     - :class:`L2SymmetricEntropicAffinity <torchdr.L2SymmetricEntropicAffinity>`
     - :class:`StudentAffinity <torchdr.StudentAffinity>`


References
----------

.. [1] Geoffrey Hinton, Sam Roweis (2002). `Stochastic Neighbor Embedding <https://proceedings.neurips.cc/paper_files/paper/2002/file/6150ccc6069bea6b5716254057a194ef-Paper.pdf>`_. Advances in Neural Information Processing Systems 15 (NeurIPS).

.. [2] Laurens van der Maaten, Geoffrey Hinton (2008). `Visualizing Data using t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl>`_. The Journal of Machine Learning Research 9.11 (JMLR).

.. [3] Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023). `SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities <https://proceedings.neurips.cc/paper_files/paper/2023/file/8b54ecd9823fff6d37e61ece8f87e534-Paper-Conference.pdf>`_. Advances in Neural Information Processing Systems 36 (NeurIPS).

.. [8] Leland McInnes, John Healy, James Melville (2018). `UMAP: Uniform manifold approximation and projection for dimension reduction <https://arxiv.org/abs/1802.03426>`_. arXiv preprint arXiv:1802.03426.

.. [13] Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). `Visualizing Large-Scale and High-Dimensional Data <https://dl.acm.org/doi/pdf/10.1145/2872427.2883041?casa_token=9ybi1tW9opcAAAAA:yVfVBu47DYa5_cpmJnQZm4PPWaTdVJgRu2pIMqm3nvNrZV5wEsM9pde03fCWixTX0_AlT-E7D3QRZw>`_. In Proceedings of the 25th international conference on world wide web.

.. [15] Sebastian Damrich, Jan Niklas Böhm, Fred Hamprecht, Dmitry Kobak (2023). `From t-SNE to UMAP with contrastive learning <https://openreview.net/pdf?id=B8a1FcY0vi>`_. International Conference on Learning Representations (ICLR).

