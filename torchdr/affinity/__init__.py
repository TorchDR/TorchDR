# -*- coding: utf-8 -*-
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from .base import Affinity, LogAffinity

from .simple import (
    ScalarProductAffinity,
    GibbsAffinity,
    SelfTuningGibbsAffinity,
    StudentAffinity,
)

from .entropic import (
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    SinkhornAffinity,
)

from .quadratic import DoublyStochasticQuadraticAffinity

from .umap import UMAPAffinityIn, UMAPAffinityOut

__all__ = [
    "Affinity",
    "LogAffinity",
    "ScalarProductAffinity",
    "GibbsAffinity",
    "SelfTuningGibbsAffinity",
    "StudentAffinity",
    "EntropicAffinity",
    "L2SymmetricEntropicAffinity",
    "SymmetricEntropicAffinity",
    "SinkhornAffinity",
    "DoublyStochasticQuadraticAffinity",
    "UMAPAffinityIn",
    "UMAPAffinityOut",
]
