# -*- coding: utf-8 -*-
# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from ._optim import binary_search, false_position, OPTIMIZERS

from ._wrappers import wrap_vectors, sum_matrix_vector

from ._geometry import pairwise_distances, LIST_METRICS

from ._validation import (
    check_NaNs,
    check_marginal,
    check_symmetry,
    check_equality_torch_keops,
    check_entropy,
)

from ._operators import entropy, kmin, kmax


__all__ = [
    "binary_search",
    "false_position",
    "OPTIMIZERS",
    "wrap_vectors",
    "kmin",
    "kmax",
    "sum_matrix_vector",
    "check_NaNs",
    "pairwise_distances",
    "LIST_METRICS",
    "check_marginal",
    "check_symmetry",
    "check_equality_torch_keops",
    "check_entropy",
    "entropy",
]
