Version 0.3 (2025-07-15)
------------------------

- Improve UMAP via direct gradient computation and edge masking `PR #198 <https://github.com/TorchDR/TorchDR/pull/198>`_.
- Support for torch.compile `PR #194 <https://github.com/TorchDR/TorchDR/pull/194>`_.
- Automatically handle duplicates `PR #188 <https://github.com/TorchDR/TorchDR/pull/188>`_.
- Standardize logging `PR #187 <https://github.com/TorchDR/TorchDR/pull/187>`_.
- Make affinity_out optional in AffinityMatcher `PR #186 <https://github.com/TorchDR/TorchDR/pull/186>`_.
- Implement PHATE algorithm `PR #185 <https://github.com/TorchDR/TorchDR/pull/185>`_.
- Implement PACMAP algorithm `PR #182 <https://github.com/TorchDR/TorchDR/pull/182>`_.
- COSNE support for hyperbolic embeddings `PR #178 <https://github.com/TorchDR/TorchDR/pull/178>`_.
- Allow for any Torch optimizer or scheduler `PR #174 <https://github.com/TorchDR/TorchDR/pull/174>`_.
- Ensure compatibility with python 3.8+ `PR #173 <https://github.com/TorchDR/TorchDR/pull/173>`_.


Version 0.2 (2025-02-07)
------------------------

- FAISS support for KNN `PR #160 <https://github.com/TorchDR/TorchDR/pull/160>`_.
- CIFAR examples with DINOv2 features `PR #158 <https://github.com/TorchDR/TorchDR/pull/158>`_.
- Fast linter and formatter with Ruff `PR #151 <https://github.com/TorchDR/TorchDR/pull/151>`_.
- Pre-commit hooks added for code quality and consistency checks `PR #147 <https://github.com/TorchDR/TorchDR/pull/147>`_.
- Incremental PCA `PR #137 <https://github.com/TorchDR/TorchDR/pull/137>`_.
- Clean citation style via sphinxcontrib-bibtex `PR #143 <https://github.com/TorchDR/TorchDR/pull/143>`_.
- Functionality to switch to keops backend if it is installed and an out-of-memory error is raised `PR #130 <https://github.com/TorchDR/TorchDR/pull/130>`_.
- Code of conduct `PR #127 <https://github.com/TorchDR/TorchDR/pull/127>`_.
- Pull request template `PR #125 <https://github.com/TorchDR/TorchDR/pull/125>`_.


Version 0.1 (2024-09-17)
------------------------

- Multiple basic affinities, including scalar product, Gaussian, and Student kernels.
- Affinities based on k-NN normalizations such as Self-tuning affinities and MAGIC.
- Doubly stochastic affinities with entropic and quadratic projections.
- Adaptive affinities with entropy control (*entropic affinity*) and its symmetric version.
- Input and output affinities of UMAP.
- A template object *AffinityMatcher* to solve DR with gradient descent and any input and output affinities.
- Neighbor embedding methods like SNE, t-SNE, t-SNEkhorn, UMAP, LargeVis, and InfoTSNE.
- Template objects for neighbor embedding methods.
- Spectral embeddings via eigendecomposition of the input affinity matrix (when applicable).
- KeOps compatibility for all components, except spectral embeddings.
- Silhouette score.
