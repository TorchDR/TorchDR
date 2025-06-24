"""Affinity matrices used in UMAP."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import math
from typing import Union, Any, Optional

import numpy as np
import torch
from scipy.optimize import curve_fit

from torchdr.affinity.base import SparseLogAffinity, UnnormalizedLogAffinity
from torchdr.utils import binary_search, kmin, wrap_vectors, check_neighbor_param


@wrap_vectors
def _log_Pumap(C, rho, sigma):
    r"""Return the log of the input affinity matrix used in UMAP."""
    return -(C - rho) / sigma


# from umap/umap/umap_.py
def find_ab_params(spread, min_dist):
    """Fit a, b params as in UMAP.

    Fit (a, b) for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class UMAPAffinityIn(SparseLogAffinity):
    r"""Compute the input affinity used in UMAP :cite:`mcinnes2018umap`.

    The algorithm computes via root search the variable
    :math:`\mathbf{\sigma}^* \in \mathbb{R}^n_{>0}` such that

    .. math::
        \forall (i,j), \: P_{ij} = \exp(- (C_{ij} - \rho_i) / \sigma^\star_i) \quad \text{where} \quad \forall i, \: \sum_j P_{ij} = \log (\mathrm{n_neighbors})

    and :math:`\rho_i = \min_j C_{ij}`.

    Parameters
    ----------
    n_neighbors : float, optional
        Number of effective nearest neighbors to consider. Similar to the perplexity.
    tol : float, optional
        Precision threshold for the root search.
    max_iter : int, optional
        Maximum number of iterations for the root search.
    sparsity : bool, optional
        Whether to use sparsity mode.
        Default is True.
    metric : str, optional
        Metric to use for pairwise distances computation.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 30,
        tol: float = 1e-5,
        max_iter: int = 1000,
        sparsity: bool = True,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.tol = tol
        self.max_iter = max_iter

        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=sparsity,
        )

    def _compute_sparse_log_affinity(self, X: Union[torch.Tensor, np.ndarray]):
        r"""Compute the input affinity matrix of UMAP from input data X.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : UMAPAffinityIn
            The fitted instance.
        """
        if self.verbose:
            self.logger.info("Computing UMAP affinity.")

        n_samples_in = X.shape[0]
        n_neighbors = check_neighbor_param(self.n_neighbors, n_samples_in)

        if self.sparsity:
            if self.verbose:
                self.logger.info(
                    f"Affinity : sparsity mode enabled, computing {n_neighbors} nearest neighbors."
                )
            # when using sparsity, we construct a reduced distance matrix
            # of shape (n_samples, n_neighbors)
            C_, indices = self._distance_matrix(X, k=n_neighbors)
        else:
            C_, indices = self._distance_matrix(X)

        self.rho_ = kmin(C_, k=1, dim=1)[0].squeeze().contiguous()

        def marginal_gap(eps):  # function to find the root of
            marg = _log_Pumap(C_, self.rho_, eps).logsumexp(1).exp().squeeze()
            return marg - math.log(n_neighbors)

        self.eps_ = binary_search(
            f=marginal_gap,
            n=n_samples_in,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            dtype=X.dtype,
            device=X.device,
        )

        log_affinity_matrix = _log_Pumap(C_, self.rho_, self.eps_)

        return log_affinity_matrix, indices


class UMAPAffinityOut(UnnormalizedLogAffinity):
    r"""Compute the affinity used in embedding space in UMAP :cite:`mcinnes2018umap`.

    Its :math:`(i,j)` coefficient is as follows:

    .. math::
        1 / \left(1 + a C_{ij}^{b} \right)

    where parameters a and b are fitted to the spread and min_dist parameters.

    Parameters
    ----------
    min_dist : float, optional
        min_dist parameter from UMAP. Provides the minimum distance apart that
        points are allowed to be.
    spread : float, optional
        spread parameter from UMAP.
    a : float, optional
        factor of the cost matrix.
    b : float, optional
        exponent of the cost matrix.
    degrees_of_freedom : int, optional
        Degrees of freedom for the Student-t distribution.
    metric : str, optional
        Metric to use for pairwise distances computation.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity. Default is False.
    """

    def __init__(
        self,
        min_dist: float = 0.1,
        spread: float = 1,
        a: Optional[float] = None,
        b: Optional[float] = None,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.min_dist = min_dist
        self.spread = spread

        if a is None or b is None:
            fitted_a, fitted_b = find_ab_params(self.spread, self.min_dist)
            self._a, self._b = fitted_a.item(), fitted_b.item()
        else:
            self._a = a
            self._b = b

    def _log_affinity_formula(self, C: Union[torch.Tensor, Any]):
        return -(1 + self._a * C**self._b).log()
