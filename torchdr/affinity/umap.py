# -*- coding: utf-8 -*-
"""
Affinity matrices used in UMAP.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import numpy as np
from scipy.optimize import curve_fit

from torchdr.affinity import Affinity
from torchdr.utils import false_position, kmin, wrap_vectors


@wrap_vectors
def _log_Pumap(C, rho, sigma):
    return -(C - rho) / sigma


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


class UMAPAffinityData(Affinity):
    def __init__(
        self,
        n_neighbors=30,  # analog of the perplexity parameter of SNE / TSNE
        tol=1e-5,
        max_iter=1000,
        sparsity=None,
        metric="euclidean",
        device=None,
        keops=True,
        verbose=True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.n_neighbors = n_neighbors
        self.tol = tol
        self.max_iter = max_iter
        self.sparsity = self.n_neighbors < 100 if sparsity is None else sparsity

    def fit(self, X):
        r"""Computes the input affinity matrix of UMAP from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
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

        C_full = self._ground_cost_matrix(self.data_)

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

        if not 1 < self.n_neighbors <= n:
            raise ValueError(
                "[TorchDR] Affinity : The k parameter must be between "
                "2 and number of samples."
            )

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


class UMAPAffinityEmbedding(Affinity):
    def __init__(
        self,
        min_dist=0.1,
        spread=1,
        a=None,
        b=None,
        metric="euclidean",
        device=None,
        keops=True,
        verbose=True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.min_dist = min_dist
        self.spread = spread

        if a is None or b is None:
            fitted_a, fitted_b = find_ab_params(self.spread, self.min_dist)
            self._a, self._b = fitted_a.item(), fitted_b.item()
        else:
            self._a = a
            self._b = b

    def fit(self, X):
        r"""
        Computes the embedding affinity matrix of UMAP from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : UMAPAffinityEmbedding
            The fitted instance.
        """
        super().fit(X)

        C = self._ground_cost_matrix(self.data_)
        self.affinity_matrix_ = 1 / (1 + self._a * C ** (2 * self._b))

        return self
