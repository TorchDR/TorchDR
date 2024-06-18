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
    GibbsAffinity,
    StudentAffinity,
    ScalarProductAffinity,
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    DoublyStochasticEntropic,
    DoublyStochasticQuadratic,
)

# import DR methods
from .spectral import PCA
from .affinity_matcher import (
    AffinityMatcher,
    BatchedAffinityMatcher,
)
from .neighbor_embedding import (
    SNE,
    TSNE,
)

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__license__",
    "GibbsAffinity",
    "StudentAffinity",
    "ScalarProductAffinity",
    "EntropicAffinity",
    "L2SymmetricEntropicAffinity",
    "SymmetricEntropicAffinity",
    "DoublyStochasticEntropic",
    "DoublyStochasticQuadratic",
    "AffinityMatcher",
    "BatchedAffinityMatcher",
    "PCA",
    "SNE",
    "TSNE",
]
