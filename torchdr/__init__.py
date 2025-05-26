# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from .__about__ import (
    __author__,
    __license__,
    __summary__,
    __title__,
    __url__,
    __version__,
)

# import affinities
from .affinity import (
    Affinity,
    DoublyStochasticQuadraticAffinity,
    EntropicAffinity,
    GaussianAffinity,
    LogAffinity,
    MAGICAffinity,
    NormalizedGaussianAffinity,
    NormalizedStudentAffinity,
    ScalarProductAffinity,
    SelfTuningAffinity,
    SinkhornAffinity,
    StudentAffinity,
    SymmetricEntropicAffinity,
    UMAPAffinityIn,
    UMAPAffinityOut,
)
from .affinity_matcher import AffinityMatcher

# import DR methods
from .base import DRModule
from .clustering import KMeans
from .eval import silhouette_samples, silhouette_score
from .neighbor_embedding import (
    SNE,
    TSNE,
    UMAP,
    InfoTSNE,
    LargeVis,
    NeighborEmbedding,
    SampledNeighborEmbedding,
    SparseNeighborEmbedding,
    TSNEkhorn,
)
from .spectral import PCA, IncrementalPCA, KernelPCA

# import utils
from .utils import binary_search, false_position, pairwise_distances

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__license__",
    "Affinity",
    "LogAffinity",
    "GaussianAffinity",
    "NormalizedGaussianAffinity",
    "NormalizedStudentAffinity",
    "SelfTuningAffinity",
    "MAGICAffinity",
    "StudentAffinity",
    "ScalarProductAffinity",
    "EntropicAffinity",
    "SymmetricEntropicAffinity",
    "SinkhornAffinity",
    "DoublyStochasticQuadraticAffinity",
    "UMAPAffinityIn",
    "UMAPAffinityOut",
    "DRModule",
    "KMeans",
    "AffinityMatcher",
    "BatchedAffinityMatcher",
    "PCA",
    "KernelPCA",
    "IncrementalPCA",
    "NeighborEmbedding",
    "SparseNeighborEmbedding",
    "SampledNeighborEmbedding",
    "SNE",
    "TSNE",
    "InfoTSNE",
    "TSNEkhorn",
    "LargeVis",
    "UMAP",
    "pairwise_distances",
    "binary_search",
    "false_position",
    "silhouette_samples",
    "silhouette_score",
]
