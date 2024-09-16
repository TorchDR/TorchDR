# -*- coding: utf-8 -*-
"""Affinity matrices used in UMAP."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from ..utils import LazyTensorType
import math
import numpy as np
from scipy.optimize import curve_fit
import warnings

from torchdr.affinity.base import UnnormalizedAffinity, SparseLogAffinity
from torchdr.utils import (
    false_position,
    kmin,
    wrap_vectors,
)


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


def _check_n_neighbors(n_neighbors, n, verbose=True):
    r"""Check the n_neighbors parameter and returns a valid value."""
    if n <= 1:
        raise ValueError(
            f"[TorchDR] ERROR : Input has less than one sample : n_samples = {n}."
        )

    if n_neighbors >= n - 1 or n_neighbors <= 1:
        new_value = n // 2
        if verbose:
            warnings.warn(
                "[TorchDR] WARNING : The n_neighbors parameter must be greater than "
                f"1 and smaller than the number of samples - 1 (here {n-1}). "
                f"Got n_neighbors = {n_neighbors}. Setting n_neighbors to {new_value}."
            )
        return new_value
    else:
        return n_neighbors


class UMAPAffinityIn(SparseLogAffinity):
    r"""Compute the input affinity used in UMAP [8]_.

    The algorithm computes via root search the variable
    :math:`\mathbf{\sigma}^* \in \mathbb{R}^n_{>0}` such that

    .. math::
        \forall i, \: \sum_j P_{ij} = \log (\mathrm{n_neighbors}) \quad \text{where} \quad \forall (i,j), \: P_{ij} = \exp(- (C_{ij} - \rho_i) / \sigma^\star_i)

    and :math:`\rho_i = \min_j C_{ij}`.

    Parameters
    ----------
    n_neighbors : float, optional
        Number of effective nearest neighbors to consider. Similar to the perplexity.
    tol : float, optional
        Precision threshold for the root search.
    max_iter : int, optional
        Maximum number of iterations for the root search.
    sparsity : bool or 'auto', optional
        Whether to use sparsity mode.
    metric : str, optional
        Metric to use for pairwise distances computation.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity. Default is False.

    References
    ----------
    .. [8] Leland McInnes, John Healy, James Melville (2018).
        UMAP: Uniform manifold approximation and projection for dimension reduction.
        arXiv preprint arXiv:1802.03426.

    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 30,  # analog of the perplexity parameter of SNE / TSNE
        tol: float = 1e-5,
        max_iter: int = 1000,
        sparsity: bool | str = "auto",
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.tol = tol
        self.max_iter = max_iter

        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
            sparsity=sparsity,
        )

    def _sparsity_rule(self):
        if self.n_neighbors < 100:
            return True
        else:
            if self.verbose:
                warnings.warn(
                    "[TorchDR] WARNING Affinity: n_neighbors is large "
                    f"({self.n_neighbors}) thus we turn off sparsity for "
                    "the EntropicAffinity. "
                )
            return False

    def _compute_sparse_log_affinity(self, X: torch.Tensor | np.ndarray):
        r"""Compute the input affinity matrix of UMAP from input data X.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : UMAPAffinityData
            The fitted instance.
        """
        if self.verbose:
            print("[TorchDR] Affinity : computing the input affinity matrix of UMAP.")

        C = self._distance_matrix(X)

        n_samples_in = C.shape[0]
        n_neighbors = _check_n_neighbors(self.n_neighbors, n_samples_in, self.verbose)

        if self._sparsity:
            if self.verbose:
                print(
                    "[TorchDR] Affinity : sparsity mode enabled, computing "
                    "nearest neighbors."
                )
            # when using sparsity, we construct a reduced distance matrix
            # of shape (n_samples, n_neighbors)
            C_, indices = kmin(C, k=n_neighbors, dim=1)
        else:
            C_, indices = C, None

        self.rho_ = kmin(C_, k=1, dim=1)[0].squeeze().contiguous()

        def marginal_gap(eps):  # function to find the root of
            marg = _log_Pumap(C_, self.rho_, eps).logsumexp(1).exp().squeeze()
            return marg - math.log(n_neighbors)

        self.eps_ = false_position(
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


class UMAPAffinityOut(UnnormalizedAffinity):
    r"""Compute the affinity used in embedding space in UMAP [8]_.

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
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity. Default is False.

    References
    ----------
    .. [8] Leland McInnes, John Healy, James Melville (2018).
        UMAP: Uniform manifold approximation and projection for dimension reduction.
        arXiv preprint arXiv:1802.03426.
    """

    def __init__(
        self,
        min_dist: float = 0.1,
        spread: float = 1,
        a: float = None,
        b: float = None,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
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

    def _affinity_formula(self, C: torch.Tensor | LazyTensorType):
        return 1 / (1 + self._a * C**self._b)
