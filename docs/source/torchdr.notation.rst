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
    - Symmetric pairwise distance matrix between the samples.
  * - :math:`\langle \mathbf{C}, \mathbf{P} \rangle = \sum_{i,j} C_{ij} P_{ij}`
    - Frobenius inner product between two matrices.
  * - :math:`\mathrm{h}(\mathbf{p}) = - \sum_{i} p_{i} (\log p_{i} - 1)`
    - Shannon entropy for vector.
  * - :math:`\mathbf{1} = (1,...,1)^\top`
    - :math:`n`-dimensional all-ones vector.
  
