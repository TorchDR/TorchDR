# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License


from .base import (
    Affinity,
    LogAffinity,
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
from .knn_normalized import MAGICAffinity, SelfTuningAffinity
from .quadratic import DoublyStochasticQuadraticAffinity
from .umap import UMAPAffinityIn, UMAPAffinityOut
from .unnormalized import (
    GaussianAffinity,
    ScalarProductAffinity,
    StudentAffinity,
    CauchyAffinity,
)

__all__ = [
    "Affinity",
    "LogAffinity",
    "UnnormalizedAffinity",
    "UnnormalizedLogAffinity",
    "SparseLogAffinity",
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
    "UMAPAffinityIn",
    "UMAPAffinityOut",
]
