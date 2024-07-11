# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
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
    SelfTuningAffinity,
    StudentAffinity,
    CauchyAffinity,
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
from .spectral import PCA
from .affinity_matcher import (
    AffinityMatcher,
)
from .neighbor_embedding import (
    SNE,
    TSNE,
    COSNE,
    InfoTSNE,
    SNEkhorn,
    TSNEkhorn,
    LargeVis,
    UMAP,
)

# import utils
from .utils import pairwise_distances, binary_search, false_position

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
    "SelfTuningAffinity",
    "StudentAffinity",
    "CauchyAffinity",
    "ScalarProductAffinity",
    "EntropicAffinity",
    "SymmetricEntropicAffinity",
    "SinkhornAffinity",
    "DoublyStochasticQuadraticAffinity",
    "UMAPAffinityIn",
    "UMAPAffinityOut",
    "DRModule",
    "AffinityMatcher",
    "BatchedAffinityMatcher",
    "PCA",
    "SNE",
    "TSNE",
    "InfoTSNE",
    "SNEkhorn",
    "TSNEkhorn",
    "LargeVis",
    "UMAP",
    "pairwise_distances",
    "binary_search",
    "false_position",
]
