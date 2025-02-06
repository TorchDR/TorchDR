
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
