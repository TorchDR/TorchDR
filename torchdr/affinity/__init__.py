# -*- coding: utf-8 -*-
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from .base import (
    Affinity,
    LogAffinity,
    SparseLogAffinity,
    UnnormalizedAffinity,
    UnnormalizedLogAffinity,
)

from .unnormalized import (
    ScalarProductAffinity,
    GaussianAffinity,
    StudentAffinity,
    CauchyAffinity,
)

from .knn_normalized import SelfTuningAffinity, MAGICAffinity

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
    "UnnormalizedAffinity",
    "UnnormalizedLogAffinity",
    "SparseLogAffinity",
    "ScalarProductAffinity",
    "GaussianAffinity",
    "NormalizedGaussianAffinity",
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
