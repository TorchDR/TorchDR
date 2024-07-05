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
    GibbsAffinity,
    NormalizedGibbsAffinity,
    SelfTuningGibbsAffinity,
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
from .spectral import PCA
from .affinity_matcher import (
    AffinityMatcher,
    BatchedAffinityMatcher,
)
from .neighbor_embedding import (
    SNE,
    TSNE,
    InfoTSNE,
    SNEkhorn,
    TSNEkhorn,
    LargeVis,
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
    "GibbsAffinity",
    "NormalizedGibbsAffinity",
    "SelfTuningGibbsAffinity",
    "StudentAffinity",
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
    "pairwise_distances",
    "binary_search",
    "false_position",
]
