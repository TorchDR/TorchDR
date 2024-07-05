# -*- coding: utf-8 -*-
"""
Affinity matrices used in UMAP.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from scipy.optimize import curve_fit

from torchdr.affinity import Affinity
from torchdr.utils import (
    false_position,
    kmin,
    wrap_vectors,
    batch_transpose,
    inputs_to_torch,
)


@wrap_vectors
def _log_Pumap(C, rho, sigma):
    r"""
    Returns the log of the input affinity matrix used in UMAP.
    """
    return -(C - rho) / sigma


def _Student_umap(C, a, b):
    r"""
    Returns the Student affinity function used in UMAP.
    """
    return 1 / (1 + a * C ** (2 * b))


# from umap/umap/umap_.py
def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
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
    r"""
    Checks the n_neighbors parameter and returns a valid value.
    """
    if n <= 1:
        raise ValueError(
            f"[TorchDR] ERROR : Input has less than one sample : n_samples = {n}."
        )

    if n_neighbors >= n or n_neighbors <= 1:
        new_value = n // 2
        if verbose:
            print(
                "[TorchDR] WARNING : The n_neighbors parameter must be greater than "
                f"1 and smaller than the number of samples (here n = {n}). "
                f"Got n_neighbors = {n_neighbors}. Setting n_neighbors to {new_value}."
            )
        return new_value
    else:
        return n_neighbors


class UMAPAffinityIn(Affinity):
    def __init__(
        self,
        n_neighbors: float = 30,  # analog of the perplexity parameter of SNE / TSNE
        tol: float = 1e-5,
        max_iter: int = 1000,
        sparsity: bool = None,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        self.n_neighbors = n_neighbors
        self.tol = tol
        self.max_iter = max_iter
        self.sparsity = self.n_neighbors < 100 if sparsity is None else sparsity

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""Computes the input affinity matrix of UMAP from input data X.

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
            print(
                "[TorchDR] Affinity : Computing the Doubly Stochastic Quadratic "
                "Affinity matrix."
            )

        super().fit(X)

        C_full = self._pairwise_distance_matrix(self.data_)

        if self.sparsity:
            print(
                "[TorchDR] Affinity : Sparsity mode enabled, computing "
                "nearest neighbors."
            )
            # when using sparsity, we construct a reduced distance matrix
            # of shape (n_samples, n_neighbors)
            C_reduced, self.indices_ = kmin(C_full, k=self.n_neighbors, dim=1)
        else:
            C_reduced = C_full

        self.rho_ = kmin(C_reduced, k=1, dim=1)[0].squeeze().contiguous()

        n = C_full.shape[0]
        self.n_neighbors = _check_n_neighbors(self.n_neighbors, n, self.verbose)

        def marginal_gap(eps):  # function to find the root of
            marg = _log_Pumap(C_reduced, self.rho_, eps).logsumexp(1).exp().squeeze()
            return marg - np.log(self.n_neighbors)

        self.eps_ = false_position(
            f=marginal_gap,
            n=n,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            dtype=self.data_.dtype,
            device=self.data_.device,
        )

        P = _log_Pumap(C_full, self.rho_, self.eps_).exp()
        self.affinity_matrix_ = P + P.T - P * P.T  # symmetrize the affinity matrix

        return self

    def get_batch(self, indices: torch.Tensor):
        r"""
        Extracts the affinity submatrix corresponding to the batch indices.

        Parameters
        ----------
        indices : torch.Tensor of shape (n_batch, batch_size)
            Indices of the batch.

        Returns
        -------
        P_batch : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_batch, batch_size, batch_size)
            The affinity matrix for the batch indices.
        """
        C_batch = super().get_batch(indices)
        rho_batch = self.rho_[indices]
        eps_batch = self.eps_[indices]
        P_batch = _log_Pumap(C_batch, rho_batch, eps_batch).exp()
        P_batch_t = batch_transpose(P_batch)
        return P_batch + P_batch_t - P_batch * P_batch_t


class UMAPAffinityOut(Affinity):
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
        verbose: bool = True,
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

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Computes the embedding affinity matrix of UMAP from input data X.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : UMAPAffinityEmbedding
            The fitted instance.
        """
        super().fit(X)

        C = self._pairwise_distance_matrix(self.data_)
        self.affinity_matrix_ = _Student_umap(C, self._a, self._b)

        return self

    def get_batch(self, indices: torch.Tensor):
        r"""
        Extracts the affinity submatrix corresponding to the batch indices.

        Parameters
        ----------
        indices : torch.Tensor of shape (n_batch, batch_size)
            Indices of the batch.

        Returns
        -------
        P_batch : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_batch, batch_size, batch_size)
            The affinity matrix for the batch indices.
        """
        C_batch = super().get_batch(indices)
        return _Student_umap(C_batch, self._a, self._b)

    @inputs_to_torch
    def transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
    ):
        r"""
        Computes the affinity between X and Y.
        If Y is None, computes the affinity between X and itself.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data.
        Y : torch.Tensor or np.ndarray
            Second Input data. Default is None.

        Returns
        -------
        P : torch.Tensor or pykeops.torch.LazyTensor
            Affinity between X and Y.
        """
        C = self._pairwise_distance_matrix(X, Y)
        return _Student_umap(C, self._a, self._b)
