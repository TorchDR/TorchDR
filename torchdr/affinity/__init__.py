# -*- coding: utf-8 -*-
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from .base import Affinity, LogAffinity

from .simple import ScalarProductAffinity, GibbsAffinity, StudentAffinity

from .entropic import (
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    DoublyStochasticEntropic,
)

from .quadratic import DoublyStochasticQuadratic

from .umap import UMAPAffinityData, UMAPAffinityEmbedding

__all__ = [
    "Affinity",
    "LogAffinity",
    "ScalarProductAffinity",
    "GibbsAffinity",
    "StudentAffinity",
    "EntropicAffinity",
    "L2SymmetricEntropicAffinity",
    "SymmetricEntropicAffinity",
    "DoublyStochasticEntropic",
    "DoublyStochasticQuadratic",
    "UMAPAffinityData",
    "UMAPAffinityEmbedding",
]
