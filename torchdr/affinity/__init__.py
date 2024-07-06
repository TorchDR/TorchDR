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
    GibbsAffinity,
    StudentAffinity,
)

from .knn_normalized import SelfTuningGibbsAffinity

from .entropic import (
    EntropicAffinity,
    SymmetricEntropicAffinity,
    SinkhornAffinity,
    NormalizedGibbsAffinity,
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
    "GibbsAffinity",
    "NormalizedGibbsAffinity",
    "SelfTuningGibbsAffinity",
    "StudentAffinity",
    "EntropicAffinity",
    "SymmetricEntropicAffinity",
    "SinkhornAffinity",
    "DoublyStochasticQuadraticAffinity",
    "UMAPAffinityIn",
    "UMAPAffinityOut",
]
