Math notations
==============

The documentation of ``torchdr`` uses the following mathematical notations.

.. list-table::
   :widths: 10 50
   :header-rows: 1
   
   * - Symbol
     - Meaning
   * - :math:`\mathbf{X} = (\mathbf{x}_1, ..., \mathbf{x}_n)^\top \in \mathbb{R}^{n \times p}`
     - Input data with :math:`n` samples and dimensionality :math:`p`.
   * - :math:`\mathbf{Z} = (\mathbf{z}_1, ..., \mathbf{z}_{n'})^\top \in \mathbb{R}^{n' \times d}`
     - Embeddings of dimensionality :math:`d < p`. In usual DR settings, :math:`n' = n`.
