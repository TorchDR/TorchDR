# -*- coding: utf-8 -*-
"""
Affinity matrices with quadratic constraints
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from tqdm import tqdm

from torchdr.affinity import Affinity
from torchdr.utils import OPTIMIZERS, wrap_vectors, check_NaNs


@wrap_vectors
def Pds(C, dual, eps):
    r"""
    Returns the quadratic doubly stochastic matrix P
    from the dual variable f and log_K = -C / eps.
    """
    return (dual + dual.T - C).clamp(0, float("inf")) / eps


class DoublyStochasticQuadratic(Affinity):
    def __init__(
        self,
        eps=1.0,
        init_dual=None,
        tol=1e-5,
        max_iter=1000,
        optimizer="Adam",
        lr=1e0,
        student=False,
        tolog=False,
        metric="euclidean",
        device=None,
        keops=True,
        verbose=True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.eps = eps
        self.init_dual = init_dual
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.lr = lr
        self.student = student
        self.tolog = tolog

    def fit(self, X):
        r"""Computes the quadratic doubly stochastic affinity matrix from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : DoublyStochasticQuadratic
            The fitted instance.
        """
        if self.verbose:
            print(
                "[TorchDR] Affinity : Computing the Doubly Stochastic Quadratic "
                "Affinity matrix."
            )

        super().fit(X)

        C = self._ground_cost_matrix(self.X_)

        n = C.shape[0]
        one = torch.ones(n, dtype=self.X_.dtype, device=self.X_.device)

        # Performs warm-start if an initial dual variable is provided
        self.dual_ = (
            torch.ones(n, dtype=self.X_.dtype, device=self.X_.device)
            if self.init_dual is None
            else self.init_dual
        )
        if self.tolog:
            self.log["dual"] = [self.dual_.clone().detach().cpu()]

        optimizer = OPTIMIZERS[self.optimizer]([self.dual_], lr=self.lr)

        # Dual ascent iterations
        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for k in pbar:
            with torch.no_grad():
                P = Pds(C, self.dual_, self.eps)
                P_sum = P.sum(1).squeeze()
                grad_dual = P_sum - one
                self.dual_.grad = grad_dual
                optimizer.step()

            check_NaNs(
                [self.dual_],
                msg=(
                    f"[TorchDR] Affinity (ERROR): NaN at iter {k}, "
                    "consider decreasing the learning rate."
                ),
            )

            if self.tolog:
                self.log["dual"].append(self.dual_.clone().detach().cpu())

            if self.verbose:
                pbar.set_description(
                    f"MARGINAL:{float(P_sum.mean().item()): .2e} "
                    f"(std:{float(P_sum.std().item()): .2e})"
                )

            if torch.norm(grad_dual) < self.tol:
                if self.verbose:
                    print(f"[TorchDR] Affinity : convergence reached at iter {k}.")
                break

            if k == self.max_iter - 1 and self.verbose:
                print(
                    "[TorchDR] Affinity (WARNING) : max iter attained, "
                    "algorithm stops but may not have converged."
                )

        self.n_iter_ = k
        self.affinity_matrix_ = Pds(C, self.dual_, self.eps)

        return self
