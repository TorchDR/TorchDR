.. _releases:

=============
Release Notes
=============

Version 0.4 (2026-XX-XX)
------------------------

Multi-GPU distributed training, CLI tool, and DataLoader integration.

Added
~~~~~

- Add TorchDR CLI for easy multi-GPU execution `PR #233 <https://github.com/TorchDR/TorchDR/pull/233>`_.
- Add DistributedContext for multi-GPU distance computations `PR #229 <https://github.com/TorchDR/TorchDR/pull/229>`_.
- Add multi-GPU affinity computations `PR #217 <https://github.com/TorchDR/TorchDR/pull/217>`_.
- Add multi-GPU neighbor embedding `PR #218 <https://github.com/TorchDR/TorchDR/pull/218>`_.
- Add distributed symmetrize sparse operations `PR #219 <https://github.com/TorchDR/TorchDR/pull/219>`_.
- Add distributed support for neighborhood preservation metric `PR #231 <https://github.com/TorchDR/TorchDR/pull/231>`_.
- Add DistributedPCA for multi-GPU PCA computation `PR #248 <https://github.com/TorchDR/TorchDR/pull/248>`_.
- Add distributed support to PCA and ExactIncrementalPCA `PR #253 <https://github.com/TorchDR/TorchDR/pull/253>`_.
- Add exact incremental PCA `PR #213 <https://github.com/TorchDR/TorchDR/pull/213>`_.
- Add k-NN label accuracy metric `PR #230 <https://github.com/TorchDR/TorchDR/pull/230>`_.
- Add k-means and neighborhood preservation evaluation scores `PR #214 <https://github.com/TorchDR/TorchDR/pull/214>`_.
- Add IVF support in FAISS k-NN backend `PR #220 <https://github.com/TorchDR/TorchDR/pull/220>`_.
- Add pairwise distances with query and key indexes `PR #216 <https://github.com/TorchDR/TorchDR/pull/216>`_.
- Add DataLoader support for memory-efficient k-NN computation `PR #236 <https://github.com/TorchDR/TorchDR/pull/236>`_.
- Add FAISS configuration options `PR #208 <https://github.com/TorchDR/TorchDR/pull/208>`_.
- Add automatic releases via GitHub Actions `PR #226 <https://github.com/TorchDR/TorchDR/pull/226>`_.
- Add cross-platform CI testing for macOS and Windows `PR #258 <https://github.com/TorchDR/TorchDR/pull/258>`_.
- Add tests for CLI, distributed, and sparse modules `PR #246 <https://github.com/TorchDR/TorchDR/pull/246>`_.

Changed
~~~~~~~

- Rename SampledNeighborEmbedding to NegativeSamplingNeighborEmbedding `PR #225 <https://github.com/TorchDR/TorchDR/pull/225>`_.
- Standardize device parameter to 'auto' default across all modules `PR #241 <https://github.com/TorchDR/TorchDR/pull/241>`_.
- Refactor device management for consistency `PR #211 <https://github.com/TorchDR/TorchDR/pull/211>`_.
- Improve memory management with torch buffers `PR #203 <https://github.com/TorchDR/TorchDR/pull/203>`_.
- Optimize root_search with masked_scatter_ `PR #204 <https://github.com/TorchDR/TorchDR/pull/204>`_.

Fixed
~~~~~

- Fix memory leak in AffinityMatcher by freeing input data after initialization `PR #223 <https://github.com/TorchDR/TorchDR/pull/223>`_.
- Fix PACMAP device handling `PR #215 <https://github.com/TorchDR/TorchDR/pull/215>`_.
- Fix AffinityMatcher kwargs mutation `PR #240 <https://github.com/TorchDR/TorchDR/pull/240>`_.
- Fix CLI port conflicts with --standalone flag `PR #251 <https://github.com/TorchDR/TorchDR/pull/251>`_.
- Fix TSNEkhorn docstring parameter name `PR #237 <https://github.com/TorchDR/TorchDR/pull/237>`_.
- Fix UMAP spectral init docstring `PR #207 <https://github.com/TorchDR/TorchDR/pull/207>`_.
- Add warning when distributed launch detected without GPU `PR #238 <https://github.com/TorchDR/TorchDR/pull/238>`_.

Removed
~~~~~~~

- Remove clustering module `PR #212 <https://github.com/TorchDR/TorchDR/pull/212>`_.
- Remove unnormalized affinity `PR #209 <https://github.com/TorchDR/TorchDR/pull/209>`_.
- Remove use_float16 parameter from FaissConfig `PR #228 <https://github.com/TorchDR/TorchDR/pull/228>`_.

Documentation
~~~~~~~~~~~~~

- Add multi-GPU distributed training documentation `PR #243 <https://github.com/TorchDR/TorchDR/pull/243>`_.
- Add DistributedPCA documentation `PR #252 <https://github.com/TorchDR/TorchDR/pull/252>`_.
- Add DataLoader feature documentation `PR #245 <https://github.com/TorchDR/TorchDR/pull/245>`_.
- Reorganize User Guide and API documentation `PR #244 <https://github.com/TorchDR/TorchDR/pull/244>`_.
- Improve FAISS installation error detection and messages `PR #227 <https://github.com/TorchDR/TorchDR/pull/227>`_.
- Fix documentation warnings and improve README `PR #239 <https://github.com/TorchDR/TorchDR/pull/239>`_.


Version 0.3 (2025-07-15)
------------------------

New algorithms (PHATE, PACMAP), torch.compile support, and Python 3.8+ compatibility.

Added
~~~~~

- Add PHATE algorithm `PR #185 <https://github.com/TorchDR/TorchDR/pull/185>`_.
- Add PACMAP algorithm `PR #182 <https://github.com/TorchDR/TorchDR/pull/182>`_.
- Add COSNE support for hyperbolic embeddings `PR #178 <https://github.com/TorchDR/TorchDR/pull/178>`_.
- Add support for torch.compile `PR #194 <https://github.com/TorchDR/TorchDR/pull/194>`_.
- Add direct gradient computation for neighbor embeddings `PR #196 <https://github.com/TorchDR/TorchDR/pull/196>`_.
- Add support for any Torch optimizer or scheduler `PR #174 <https://github.com/TorchDR/TorchDR/pull/174>`_.
- Add automatic duplicate handling `PR #188 <https://github.com/TorchDR/TorchDR/pull/188>`_.

Changed
~~~~~~~

- Optimize UMAP memory via direct gradient computation and edge masking `PR #198 <https://github.com/TorchDR/TorchDR/pull/198>`_.
- Make affinity_out optional in AffinityMatcher `PR #186 <https://github.com/TorchDR/TorchDR/pull/186>`_.
- Standardize logging across modules `PR #187 <https://github.com/TorchDR/TorchDR/pull/187>`_.
- Enhance logging with timing information `PR #193 <https://github.com/TorchDR/TorchDR/pull/193>`_.
- Align DRModule with scikit-learn API and improve type hints `PR #191 <https://github.com/TorchDR/TorchDR/pull/191>`_.
- Reorganize affinity module structure `PR #189 <https://github.com/TorchDR/TorchDR/pull/189>`_.
- Ensure compatibility with Python 3.8+ `PR #173 <https://github.com/TorchDR/TorchDR/pull/173>`_.

Fixed
~~~~~

- Fix PyKeOps and FAISS import errors `PR #177 <https://github.com/TorchDR/TorchDR/pull/177>`_.
- Fix hyperbolic optimization without geoopt dependency `PR #183 <https://github.com/TorchDR/TorchDR/pull/183>`_.
- Fix InfoTSNE docstring typo `PR #184 <https://github.com/TorchDR/TorchDR/pull/184>`_.


Version 0.2 (2025-02-07)
------------------------

FAISS support, incremental PCA, and improved developer tooling.

Added
~~~~~

- Add FAISS support for k-NN `PR #160 <https://github.com/TorchDR/TorchDR/pull/160>`_.
- Add extensible k-NN backend system `PR #159 <https://github.com/TorchDR/TorchDR/pull/159>`_.
- Add incremental PCA `PR #137 <https://github.com/TorchDR/TorchDR/pull/137>`_.
- Add automatic fallback to KeOps backend on out-of-memory errors `PR #130 <https://github.com/TorchDR/TorchDR/pull/130>`_.
- Add improved random seeding system `PR #154 <https://github.com/TorchDR/TorchDR/pull/154>`_.
- Add CIFAR examples with DINOv2 features `PR #158 <https://github.com/TorchDR/TorchDR/pull/158>`_.
- Add single-cell RNA-seq example with Census data `PR #166 <https://github.com/TorchDR/TorchDR/pull/166>`_.
- Add pre-commit hooks for code quality `PR #147 <https://github.com/TorchDR/TorchDR/pull/147>`_.
- Add code of conduct `PR #127 <https://github.com/TorchDR/TorchDR/pull/127>`_.
- Add pull request template `PR #125 <https://github.com/TorchDR/TorchDR/pull/125>`_.

Changed
~~~~~~~

- Rename tol parameter to min_grad_norm `PR #162 <https://github.com/TorchDR/TorchDR/pull/162>`_.
- Switch to Ruff for fast linting and formatting `PR #151 <https://github.com/TorchDR/TorchDR/pull/151>`_.
- Improve citation style via sphinxcontrib-bibtex `PR #143 <https://github.com/TorchDR/TorchDR/pull/143>`_.
- Set sparsity to True by default `PR #156 <https://github.com/TorchDR/TorchDR/pull/156>`_.

Fixed
~~~~~

- Fix clustering base methods `PR #126 <https://github.com/TorchDR/TorchDR/pull/126>`_.
- Fix documentation consistency with bibliography `PR #144 <https://github.com/TorchDR/TorchDR/pull/144>`_.


Version 0.1 (2024-09-17)
------------------------

Initial release with core dimensionality reduction algorithms.

Added
~~~~~

- Add basic affinities: scalar product, Gaussian, and Student kernels.
- Add k-NN normalized affinities: Self-tuning affinities and MAGIC.
- Add doubly stochastic affinities with entropic and quadratic projections.
- Add adaptive affinities with entropy control (entropic affinity) and symmetric version.
- Add UMAP input and output affinities.
- Add AffinityMatcher template for DR with gradient descent and custom affinities.
- Add batched AffinityMatcher for efficient batch processing `PR #12 <https://github.com/TorchDR/TorchDR/pull/12>`_.
- Add neighbor embedding methods: SNE, t-SNE, t-SNEkhorn, UMAP, LargeVis, InfoTSNE.
- Add SampledNeighborEmbedding for efficient negative sampling `PR #72 <https://github.com/TorchDR/TorchDR/pull/72>`_.
- Add template objects for neighbor embedding methods.
- Add spectral embeddings via eigendecomposition of input affinity matrix.
- Add Kernel PCA for nonlinear dimensionality reduction `PR #41 <https://github.com/TorchDR/TorchDR/pull/41>`_.
- Add k-means clustering `PR #111 <https://github.com/TorchDR/TorchDR/pull/111>`_.
- Add sparse affinity computations `PR #70 <https://github.com/TorchDR/TorchDR/pull/70>`_.
- Add SparseAffinityMatcher `PR #71 <https://github.com/TorchDR/TorchDR/pull/71>`_.
- Add transform method for affinities to apply to new data `PR #69 <https://github.com/TorchDR/TorchDR/pull/69>`_.
- Add callable affinity objects `PR #84 <https://github.com/TorchDR/TorchDR/pull/84>`_.
- Add pairwise symmetric distances with indices `PR #75 <https://github.com/TorchDR/TorchDR/pull/75>`_.
- Add KeOps compatibility for all components except spectral embeddings.
- Add silhouette score evaluation metric.

Changed
~~~~~~~

- Improve UMAP and LargeVis with stable attractive term `PR #89 <https://github.com/TorchDR/TorchDR/pull/89>`_.
- Support affinity initialization from precomputed arrays `PR #80 <https://github.com/TorchDR/TorchDR/pull/80>`_.
- Rename Gibbs affinity to Gaussian affinity `PR #74 <https://github.com/TorchDR/TorchDR/pull/74>`_.
- Make PyKeOps an optional dependency `PR #77 <https://github.com/TorchDR/TorchDR/pull/77>`_.

Removed
~~~~~~~

- Remove SNEkhorn in favor of TSNEkhorn `PR #90 <https://github.com/TorchDR/TorchDR/pull/90>`_.
