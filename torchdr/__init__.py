# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
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
    LogAffinity,
    SparseAffinity,
    SparseLogAffinity,
    DoublyStochasticQuadraticAffinity,
    EntropicAffinity,
    MAGICAffinity,
    NormalizedGaussianAffinity,
    NormalizedStudentAffinity,
    SelfTuningAffinity,
    SinkhornAffinity,
    SymmetricEntropicAffinity,
    UMAPAffinity,
    PHATEAffinity,
    PACMAPAffinity,
)
from .affinity_matcher import AffinityMatcher

# import DR methods
from .base import DRModule
from .eval import silhouette_samples, silhouette_score
from .neighbor_embedding import (
    SNE,
    TSNE,
    COSNE,
    UMAP,
    InfoTSNE,
    LargeVis,
    NeighborEmbedding,
    NegativeSamplingNeighborEmbedding,
    SparseNeighborEmbedding,
    TSNEkhorn,
    PACMAP,
)
from .spectral_embedding import (
    IncrementalPCA,
    ExactIncrementalPCA,
    KernelPCA,
    PCA,
    PHATE,
)

# import utils
from .utils import binary_search, false_position
from .distance import pairwise_distances

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__license__",
    "Affinity",
    "SparseAffinity",
    "SparseLogAffinity",
    "LogAffinity",
    "NormalizedGaussianAffinity",
    "NormalizedStudentAffinity",
    "SelfTuningAffinity",
    "MAGICAffinity",
    "EntropicAffinity",
    "SymmetricEntropicAffinity",
    "SinkhornAffinity",
    "DoublyStochasticQuadraticAffinity",
    "UMAPAffinity",
    "DRModule",
    "AffinityMatcher",
    "PCA",
    "KernelPCA",
    "IncrementalPCA",
    "ExactIncrementalPCA",
    "NeighborEmbedding",
    "SparseNeighborEmbedding",
    "NegativeSamplingNeighborEmbedding",
    "SNE",
    "TSNE",
    "COSNE",
    "InfoTSNE",
    "TSNEkhorn",
    "LargeVis",
    "UMAP",
    "PACMAP",
    "pairwise_distances",
    "binary_search",
    "false_position",
    "silhouette_samples",
    "silhouette_score",
    "PHATE",
    "PHATEAffinity",
    "PACMAPAffinity",
]
