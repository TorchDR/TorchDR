# -*- coding: utf-8 -*-
# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from .optim import binary_search, false_position, OPTIMIZERS

from .wrappers import (
    wrap_vectors,
    to_torch,
    torch_to_backend,
    handle_backend,
    sum_all_axis,
)

from .geometry import pairwise_distances, LIST_METRICS

from .validation import (
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

from .utils import (
    entropy,
    kmin,
    kmax,
    normalize_matrix,
    svd_flip,
    center_kernel,
    sum_matrix_vector,
    sum_red,
    logsumexp_red,
    batch_transpose,
    extract_batch_normalization,
)


__all__ = [
    "binary_search",
    "false_position",
    "OPTIMIZERS",
    "wrap_vectors",
    "kmin",
    "kmax",
    "sum_matrix_vector",
    "sum_red",
    "logsumexp_red",
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
    "normalize_matrix",
    "svd_flip",
    "center_kernel",
    "to_torch",
    "torch_to_backend",
    "handle_backend",
    "batch_transpose",
    "extract_batch_normalization",
]
