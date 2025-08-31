# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

from .keops import LazyTensor, LazyTensorType, is_lazy_tensor, pykeops
from .faiss import faiss
from .root_search import binary_search, false_position
from .utils import (
    seed_everything,
    matrix_transpose,
    center_kernel,
    cross_entropy_loss,
    entropy,
    kmax,
    kmin,
    logsumexp_red,
    prod_matrix_vector,
    square_loss,
    sum_matrix_vector,
    sum_red,
    svd_flip,
    bool_arg,
    matrix_power,
    identity_matrix,
    set_logger,
)
from .validation import (
    check_entropy,
    check_entropy_lower_bound,
    check_marginal,
    check_NaNs,
    check_nonnegativity,
    check_nonnegativity_eigenvalues,
    check_shape,
    check_similarity,
    check_similarity_torch_keops,
    check_symmetry,
    check_total_sum,
    check_type,
    check_neighbor_param,
    relative_similarity,
    validate_tensor,
)
from .wrappers import (
    handle_input_output,
    to_torch,
    restore_original_format,
    wrap_vectors,
    compile_if_requested,
)

from .manifold import (
    Manifold,
    ManifoldParameter,
    EuclideanManifold,
    PoincareBallManifold,
)
from .radam import RiemannianAdam
from .visu import plot_disk, plot_poincare_disk

__all__ = [
    "seed_everything",
    "LazyTensor",
    "LazyTensorType",
    "is_lazy_tensor",
    "pykeops",
    "binary_search",
    "false_position",
    "cross_entropy_loss",
    "square_loss",
    "wrap_vectors",
    "kmin",
    "kmax",
    "sum_matrix_vector",
    "prod_matrix_vector",
    "sum_red",
    "logsumexp_red",
    "check_NaNs",
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
    "svd_flip",
    "center_kernel",
    "to_torch",
    "restore_original_format",
    "handle_input_output",
    "matrix_transpose",
    "faiss",
    "bool_arg",
    "check_neighbor_param",
    "Manifold",
    "ManifoldParameter",
    "EuclideanManifold",
    "PoincareBallManifold",
    "RiemannianAdam",
    "matrix_power",
    "identity_matrix",
    "set_logger",
    "validate_tensor",
    "plot_disk",
    "plot_poincare_disk",
    "compile_if_requested",
]
