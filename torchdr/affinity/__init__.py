# -*- coding: utf-8 -*-
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from ._base import ScalarProductAffinity, GibbsAffinity, StudentAffinity

from ._entropic import (
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    DoublyStochasticEntropic,
    entropy,
    log_Pe,
    log_Pse,
    log_Pds,
    bounds_entropic_affinity,
)

__all__ = [
    "ScalarProductAffinity",
    "GibbsAffinity",
    "StudentAffinity",
    "EntropicAffinity",
    "L2SymmetricEntropicAffinity",
    "SymmetricEntropicAffinity",
    "DoublyStochasticEntropic",
    "entropy",
    "log_Pe",
    "log_Pse",
    "log_Pds",
    "bounds_entropic_affinity",
]
