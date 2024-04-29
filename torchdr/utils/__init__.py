# -*- coding: utf-8 -*-
# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from ._optim import binary_search, false_position, OPTIMIZERS

from ._wrappers import wrap_vectors

from ._geometry import pairwise_distances, LIST_METRICS

from ._validation import (
    check_NaNs,
    check_marginal,
    relative_similarity,
    check_similarity,
    check_symmetry,
    check_similarity_torch_keops,
    check_entropy,
    check_entropy_lower_bound,
    check_type,
    check_shape,
    check_nonnegativity,
    check_nonnegativity_eigenvalues,
    check_total_sum,
)

from ._operators import (
    entropy,
    kmin,
    kmax,
    cross_entropy_loss,
    square_loss,
    normalize_matrix,
    svd_flip,
    center_kernel,
    sum_matrix_vector,
)


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
    "relative_similarity",
    "check_similarity",
    "check_symmetry",
    "check_similarity_torch_keops",
    "check_entropy",
    "check_entropy_lower_bound",
    "check_type",
    "check_shape",
    "check_nonnegativity",
    "check_nonnegativity_eigenvalues",
    "check_total_sum",
    "entropy",
    "cross_entropy_loss",
    "square_loss",
    "normalize_matrix",
    "svd_flip",
    "center_kernel",
]
