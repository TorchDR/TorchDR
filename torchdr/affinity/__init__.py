# -*- coding: utf-8 -*-
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from .base import (
    Affinity,
    LogAffinity,
    TransformableAffinity,
    TransformableLogAffinity,
    SparseLogAffinity,
)

from .simple import (
    ScalarProductAffinity,
    GaussianAffinity,
    StudentAffinity,
    CauchyAffinity,
)

from .knn_normalized import SelfTuningAffinity

from .entropic import (
    EntropicAffinity,
    SymmetricEntropicAffinity,
    SinkhornAffinity,
    NormalizedGaussianAffinity,
)

from .quadratic import DoublyStochasticQuadraticAffinity

from .umap import UMAPAffinityIn, UMAPAffinityOut

__all__ = [
    "Affinity",
    "LogAffinity",
    "TransformableAffinity",
    "TransformableLogAffinity",
    "SparseLogAffinity",
    "ScalarProductAffinity",
    "GaussianAffinity",
    "NormalizedGaussianAffinity",
    "SelfTuningAffinity",
    "StudentAffinity",
    "CauchyAffinity",
    "EntropicAffinity",
    "SymmetricEntropicAffinity",
    "SinkhornAffinity",
    "DoublyStochasticQuadraticAffinity",
    "UMAPAffinityIn",
    "UMAPAffinityOut",
]
