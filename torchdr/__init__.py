# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from .__about__ import (
    __title__,
    __summary__,
    __version__,
    __url__,
    __author__,
    __license__,
)

# import affinities
from .affinity import (
    Affinity,
    LogAffinity,
    GaussianAffinity,
    NormalizedGaussianAffinity,
    NormalizedStudentAffinity,
    SelfTuningAffinity,
    MAGICAffinity,
    StudentAffinity,
    ScalarProductAffinity,
    EntropicAffinity,
    SymmetricEntropicAffinity,
    SinkhornAffinity,
    DoublyStochasticQuadraticAffinity,
    UMAPAffinityIn,
    UMAPAffinityOut,
)

# import DR methods
from .base import DRModule
from .spectral import PCA, KernelPCA
from .clustering import KMeans
from .affinity_matcher import (
    AffinityMatcher,
)
from .neighbor_embedding import (
    NeighborEmbedding,
    SparseNeighborEmbedding,
    SampledNeighborEmbedding,
    SNE,
    TSNE,
    InfoTSNE,
    TSNEkhorn,
    LargeVis,
    UMAP,
)

# import utils
from .utils import pairwise_distances, binary_search, false_position

from .eval import silhouette_samples, silhouette_score

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
