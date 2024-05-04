# -*- coding: utf-8 -*-
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from ._base import (
    Affinity,
    LogAffinity,
    ScalarProductAffinity,
    GibbsAffinity,
    StudentAffinity,
)

from ._entropic import (
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    DoublyStochasticEntropic,
)

from ._quadratic import DoublyStochasticQuadratic

from ._umap import UMAPAffinityData, UMAPAffinityEmbedding

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
