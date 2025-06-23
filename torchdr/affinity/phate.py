"""Affinity matrices used in PHATE."""

# Author: Guillaume Huguet @guillaumehu
#         Danqi Liao @Danqi7
#         Hugues Van Assel @huguesva
#
# License: BSD 3-Clause License

from typing import Optional

import torch

from torchdr.affinity.base import Affinity
from torchdr.utils import (
    matrix_transpose,
    kmin,
    sum_red,
    wrap_vectors,
    pairwise_distances,
)
from torchdr.utils import (
    matrix_power,
)


@wrap_vectors
def _log_P(C, sigma, alpha=1.0):
    return -((C / sigma) ** alpha)


class PHATEAffinity(Affinity):
    r"""Compute the potential affinity used in PHATE :cite:`moon2019visualizing`.

    The method follows these steps:
    1. Compute pairwise distance matrix
    2. Find k-th nearest neighbor distances to set bandwidth sigma
    3. Compute base affinity with alpha-decay kernel: exp(-((d/sigma)^alpha))
    4. Symmetrize the affinity matrix
    5. Row-normalize to create diffusion matrix
    6. Raise diffusion matrix to power t (diffusion steps)
    7. Compute potential distances from the diffused matrix
    8. Return negative potential distances as affinities

    Parameters
    ----------
    metric : str, optional (default="sqeuclidean")
        Metric to use for pairwise distances computation.
    device : str, optional (default=None)
        Device to use for computations. If None, uses the device of input data.
    backend : {"keops", "faiss", None}, optional (default=None)
        Which backend to use for handling sparsity and memory efficiency.
    verbose : bool, optional (default=False)
        Whether to print verbose output during computation.
    k : int, optional (default=5)
        Number of nearest neighbors used to determine bandwidth parameter sigma.
    alpha : float, optional (default=10.0)
        Exponent for the alpha-decay kernel in affinity computation.
    t : int, optional (default=5)
        Number of diffusion steps (power to raise diffusion matrix).
    eps : float, optional (default=1e-12)
        Small value to avoid numerical issues in logarithm computation.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        device: str = None,
        backend: Optional[str] = None,
        verbose: bool = False,
        k: int = 5,
        alpha: float = 10.0,
        t: int = 5,
        eps: float = 1e-12,
    ):
        if backend == "faiss" or backend == "keops":
            raise ValueError(
                f"[TorchDR] ERROR : {self.__class__.__name__} class does not support backend {backend}."
            )

        super().__init__(
            metric=metric,
            device=device,
            backend=backend,
            verbose=verbose,
            zero_diag=False,
        )

        self.alpha = alpha
        self.k = k
        self.t = t
        self.eps = eps

    def _compute_affinity(self, X: torch.Tensor):
        C, _ = self._distance_matrix(X)

        minK_values, _ = kmin(C, k=self.k, dim=1)
        self.sigma_ = minK_values[:, -1]
        affinity = _log_P(C, self.sigma_, self.alpha).exp()
        affinity = (affinity + matrix_transpose(affinity)) / 2
        affinity = affinity / sum_red(affinity, dim=1)
        affinity = matrix_power(affinity, self.t)
        potential_dist, _ = pairwise_distances(
            -(affinity + self.eps).log(), metric="euclidean", backend=self.backend
        )
        return -potential_dist
