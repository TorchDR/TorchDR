Math notations
==============

The documentation of ``torchdr`` uses the following mathematical notations.

.. list-table::
  :widths: 10 50
  :header-rows: 1
   
  * - Symbol
    - Meaning
  * - :math:`\mathbf{X} = (\mathbf{x}_1, ..., \mathbf{x}_n)^\top`
    - Input data with :math:`n` samples and dimensionality :math:`p`.
  * - :math:`\mathbf{Z} = (\mathbf{z}_1, ..., \mathbf{z}_{n'})^\top`
    - Embeddings. In usual DR settings, :math:`n' = n`.
  * - :math:`\mathbf{C} \in \mathbb{R}^{n \times n}` 
    - Symmetric pairwise distance matrix between the samples (depending on the context).
  * - :math:`\langle \mathbf{C}, \mathbf{P} \rangle = \sum_{i,j} C_{ij} P_{ij}`
    - Frobenius inner product between two matrices.
  * - :math:`\mathrm{h}(\mathbf{p}) = - \sum_{i} p_{i} (\log p_{i} - 1)`
    - Shannon entropy for vector.
  * - :math:`\mathrm{H}(\mathbf{P}) := - \sum_{ij} P_{ij} (\log P_{ij} - 1)`
    - Global Shannon entropy for matrices.
  * - :math:`\mathbf{1} = (1,...,1)^\top`
    - :math:`n`-dimensional all-ones vector.
  * - :math:`\mathrm{KL}(\mathbf{P} \| \mathbf{Q}) := \sum_{ij} P_{ij} (\log (Q_{ij} / P_{ij}) - 1) + Q_{ij}`
    - Kullback Leibler divergence between :math:`\mathbf{P}` and :math:`\mathbf{Q}`.
  * - :math:`\mathcal{DS} := \left\{ \mathbf{P} \in \mathbb{R}_+^{n \times n}: \: \mathbf{P} = \mathbf{P}^\top \:,\: \mathbf{P} \mathbf{1} = \mathbf{1} \right\}`
    - Set of symmetric doubly stochastic matrices.