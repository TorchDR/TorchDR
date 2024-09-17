First pre-release : 0.1
-----------------------

*17 Sep 2024*

It provides the following features:

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
