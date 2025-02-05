# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from .geometry import (
    LIST_METRICS_KEOPS,
    LIST_METRICS_FAISS,
    pairwise_distances,
    symmetric_pairwise_distances_indices,
)
from .keops import LazyTensor, LazyTensorType, is_lazy_tensor, pykeops
from .faiss import faiss
from .optim import OPTIMIZERS, binary_search, false_position
from .utils import (
    seed_everything,
    batch_transpose,
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
    relative_similarity,
)
from .wrappers import (
    handle_type,
    handle_keops,
    sum_output,
    to_torch,
    torch_to_backend,
    wrap_vectors,
)

__all__ = [
    "seed_everything",
    "LazyTensor",
    "LazyTensorType",
    "is_lazy_tensor",
    "pykeops",
    "binary_search",
    "false_position",
    "OPTIMIZERS",
    "cross_entropy_loss",
    "square_loss",
    "wrap_vectors",
    "kmin",
    "kmax",
    "sum_matrix_vector",
    "prod_matrix_vector",
    "sum_red",
    "sum_output",
    "logsumexp_red",
    "check_NaNs",
    "pairwise_distances",
    "symmetric_pairwise_distances_indices",
    "LIST_METRICS_KEOPS",
    "LIST_METRICS_FAISS",
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
    "torch_to_backend",
    "handle_type",
    "batch_transpose",
    "handle_keops",
    "faiss",
]
