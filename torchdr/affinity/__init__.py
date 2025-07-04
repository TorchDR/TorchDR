# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License


from .base import (
    Affinity,
    LogAffinity,
    SparseAffinity,
    SparseLogAffinity,
    UnnormalizedAffinity,
    UnnormalizedLogAffinity,
)
from .entropic import (
    EntropicAffinity,
    NormalizedGaussianAffinity,
    NormalizedStudentAffinity,
    SinkhornAffinity,
    SymmetricEntropicAffinity,
)
from .knn_normalized import (
    MAGICAffinity,
    SelfTuningAffinity,
    PHATEAffinity,
    PACMAPAffinity,
    UMAPAffinity,
)
from .quadratic import DoublyStochasticQuadraticAffinity
from .unnormalized import (
    GaussianAffinity,
    NegativeCostAffinity,
    ScalarProductAffinity,
    StudentAffinity,
    CauchyAffinity,
)

__all__ = [
    "Affinity",
    "SparseAffinity",
    "LogAffinity",
    "UnnormalizedAffinity",
    "UnnormalizedLogAffinity",
    "SparseLogAffinity",
    "NegativeCostAffinity",
    "ScalarProductAffinity",
    "GaussianAffinity",
    "NormalizedGaussianAffinity",
    "NormalizedStudentAffinity",
    "SelfTuningAffinity",
    "MAGICAffinity",
    "StudentAffinity",
    "CauchyAffinity",
    "EntropicAffinity",
    "SymmetricEntropicAffinity",
    "SinkhornAffinity",
    "DoublyStochasticQuadraticAffinity",
    "UMAPAffinity",
    "PACMAPAffinity",
    "PHATEAffinity",
]
