# -*- coding: utf-8 -*-
"""
Entropic projections to construct (entropic) affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from tqdm import tqdm

from torchdr.utils.optim import false_position, OPTIMIZERS
from torchdr.utils.geometry import pairwise_distances
from torchdr.utils.utils import (
    keops_support,
    sum_matrix_vector,
    kmin,
    kmax,
    check_NaNs,
)
from torchdr.affinity.base import BaseAffinity, LogAffinity


def entropy(P, log=True, dim=1):
    r"""
    Computes the entropy of P along axis dim.
    Supports log domain input.
    """
    if log:
        return -(P.exp() * (P - 1)).sum(dim).squeeze()
    else:
        return -(P * (P.log() - 1)).sum(dim).squeeze()


@keops_support
def log_Pe(C, eps):
    r"""
    Returns the log of the directed entropic affinity matrix
    with prescribed kernel bandwidth epsilon.
    """
    log_P = -C / eps
    return log_P - log_P.logsumexp(1)[:, None]


@keops_support
def log_Pse(C, eps, mu, eps_square=False):
    r"""
    Returns the log of the symmetric entropic affinity matrix
    with given (dual) variables epsilon and mu.
    """
    if eps_square:
        eps = eps**2

    return (mu + mu.T - 2 * C) / (eps + eps.T)


@keops_support
def log_Pds(log_K, f):
    r"""
    Returns the log of the doubly stochastic normalization of log_K (in log domain)
    given a scaling vector f which can be computed via the Sinkhorn iterations.
    """
    return f + f.T + log_K


def bounds_entropic_affinity(C, perplexity):
    r"""
    Computes the bounds derived in [4] for the entropic affinity root.

    Parameters
    ----------
    C : tensor or lazy tensor of shape (n_samples, n_samples)
        Distance matrix between the samples.
    perplexity : float
        Perplexity parameter, related to the number of 'effective' nearest neighbors.

    Returns
    -------
    begin : tensor of shape (n_samples)
        Lower bound of the root.
    end : tensor of shape (n_samples)
        Upper bound of the root.

    References
    ----------
    .. [4] Entropic Affinities: Properties and Efficient Numerical Computation.
        Max Vladymyrov, Miguel A. Carreira-Perpinan, ICML 2013.
    """
    N = C.shape[0]

    # we use the same notations as in [4] for clarity purposes

    # solve a unique 1D root finding problem
    def find_p1(x):
        return np.log(np.min([np.sqrt(2 * N), perplexity])) - 2 * (1 - x) * np.log(
            N / (2 * (1 - x))
        )

    begin = 3 / 4
    end = 1 - 1e-6
    p1 = false_position(
        f=find_p1, n=1, begin=begin, end=end, max_iter=1000, tol=1e-6, verbose=True
    ).item()

    # retrieve greatest and smallest pairwise distances
    dN = kmax(C, k=1, dim=1)
    d12 = kmin(C, k=2, dim=1)
    d1 = d12[:, 0]
    d2 = d12[:, 1]
    Delta_N = dN - d1
    Delta_2 = d2 - d1

    # compute bounds derived in [4]
    beta_L = torch.max(
        (N * np.log(N / perplexity)) / ((N - 1) * Delta_N),
        torch.sqrt(np.log(N / perplexity) / (dN**2 - d1**2)),
    )
    beta_U = (1 / Delta_2) * np.log((N - 1) * p1 / (1 - p1))

    # convert to our notations
    begin = 1 / beta_U
    end = 1 / beta_L

    return begin, end


class EntropicAffinity(LogAffinity):
    r"""
    Solves the (directed) entropic affinity problem introduced in [1].
    Also corresponds to the Pe matrix in [3] (see also [3]).

    For the affinity matrix of t-SNE [4], use L2SymmetricEntropicAffinity instead.
    For the affinity matrix of SNEkhorn/t-SNEkhorn [2], use SymmetricEntropicAffinity.

    Parameters
    ----------
    perplexity : float
        Perplexity parameter, related to the number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
    tol : float, optional
        Precision threshold at which the root finding algorithm stops.
    metric : str, optional
        Metric to use for computing distances (default "euclidean").
    max_iter : int, optional
        Number of maximum iterations for the root finding algorithm.
    verbose : bool, optional
        Verbosity.
    keops : bool, optional
        Whether to use KeOps for computation (default True).

    References
    ----------
    .. [1] Stochastic Neighbor Embedding.
        Geoffrey Hinton, Sam Roweis, NeurIPS 2002.
    .. [2] Visualizing Data using t-SNE.
        Laurens Van der Maaten, Geoffrey Hinton, JMLR 2008.
    .. [3] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities.
        Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    .. [4] Entropic Affinities: Properties and Efficient Numerical Computation.
        Max Vladymyrov, Miguel A. Carreira-Perpinan, ICML 2013.
    """

    def __init__(
        self,
        perplexity,
        metric="euclidean",
        tol=1e-3,
        max_iter=1000,
        verbose=True,
        keops=True,
    ):
        self.perplexity = perplexity
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.keops = keops
        super(EntropicAffinity, self).__init__()

    def fit(self, X):
        r"""
        Solves the problem (EA) in [1] to compute the entropic affinity matrix in
        log space (which is **not** symmetric).

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        References
        ----------
        .. [1] Stochastic Neighbor Embedding.
            Geoffrey Hinton, Sam Roweis, NeurIPS 2002.
        """
        super(EntropicAffinity, self).fit(X)

        C = pairwise_distances(X, metric=self.metric, keops=self.keops)

        target_entropy = np.log(self.perplexity) + 1
        n = X.shape[0]

        if not 1 < self.perplexity <= n:
            raise ValueError(
                "[TorchDR] Affinity : The perplexity parameter must be between \
                    1 and number of samples."
            )

        def entropy_gap(eps):  # function to find the root of
            return entropy(log_Pe(C, eps), log=True) - target_entropy

        begin, end = bounds_entropic_affinity(C, self.perplexity)

        self.eps_ = false_position(
            f=entropy_gap,
            n=n,
            begin=begin,
            end=end,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            dtype=X.dtype,
            device=X.device,
        )
        self.log_affinity_matrix_ = log_Pe(C, self.eps_)


class L2SymmetricEntropicAffinity(BaseAffinity):
    r"""
    Computes the L2-symmetrized entropic affinity matrix of t-SNE [2].

    Parameters
    ----------
    perplexity : float
        Perplexity parameter, related to the number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
    tol : float, optional
        Precision threshold at which the root finding algorithm stops.
    max_iter : int, optional
        Number of maximum iterations for the root finding algorithm.
    verbose : bool, optional
        Verbosity.
    keops : bool, optional
        Whether to use KeOps for computation (default True).

    References
    ----------
    .. [2] Visualizing Data using t-SNE.
        Laurens Van der Maaten, Geoffrey Hinton, JMLR 2008.
    """

    def __init__(
        self,
        perplexity,
        metric="euclidean",
        tol=1e-5,
        max_iter=1000,
        verbose=True,
        keops=True,
    ):
        self.perplexity = perplexity
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.keops = keops
        super(L2SymmetricEntropicAffinity, self).__init__()

    def fit(self, X):
        super().fit(X)

        EA = EntropicAffinity(
            perplexity=self.perplexity,
            metric=self.metric,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            keops=self.keops,
        )
        EA.fit(X)
        self.eps_ = EA.eps_

        log_P = EA.log_affinity_matrix_
        self.affinity_matrix_ = (log_P.exp() + log_P.exp().T) / 2


class SymmetricEntropicAffinity(LogAffinity):
    r"""
    Computes the solution to the symmetric entropic affinity problem described in [3],
    in log space.
    More precisely, it solves equation (SEA) in [3] with the dual ascent procedure.

    Parameters
    ----------
    perplexity : float
        Perplexity parameter, related to the number of 'effective' nearest
        neighbors. Consider selecting a value between 2 and the number of
        samples.
    lr : float, optional
        Learning rate used to update dual variables.
    eps_square : bool, optional
        Whether to optimize on the square of the dual variables.
        May be more stable in practice.
    metric : str, optional
        Metric to use for computing distances, by default "euclidean".
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-5.
    max_iter : int, optional
        Number of maximum iterations for the algorithm, by default 500.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use (default 'Adam').
    verbose : bool, optional
        Verbosity (default True).
    tolog : bool, optional
        Whether to store intermediate result in a dictionary (default False).
    keops : bool, optional
        Whether to use KeOps for computation, by default True.

    Attributes
    ----------
    log : dictionary
        Contains the loss and the dual variables at each iteration of the
        optimization algorithm when tolog = True.
    n_iter_ : int
        Number of iterations run.
    eps_ : torch.Tensor of shape (n_samples)
        Dual variable associated to the entropy constraint.
    mu_ : torch.Tensor of shape (n_samples)
        Dual variable associated to the marginal constraint.

    References
    ----------
    .. [3] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities,
        Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    """

    def __init__(
        self,
        perplexity,
        lr=1e0,
        eps_square=False,
        metric="euclidean",
        tol=1e-3,
        max_iter=500,
        optimizer="Adam",
        verbose=True,
        tolog=False,
        keops=True,
    ):
        self.perplexity = perplexity
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.verbose = verbose
        self.tolog = tolog
        self.n_iter_ = 0
        self.eps_square = eps_square
        self.metric = metric
        self.keops = keops
        super(SymmetricEntropicAffinity, self).__init__()

    def fit(self, X):
        super().fit(X)

        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        n = X.shape[0]
        if not 1 < self.perplexity <= n:
            raise ValueError(
                "[TorchDR] Affinity : The perplexity parameter must be between \
                    1 and number of samples."
            )
        target_entropy = np.log(self.perplexity) + 1

        # dual variables, size (n_samples)
        self.eps_ = torch.ones(n, dtype=X.dtype, device=X.device)
        self.mu_ = torch.zeros(n, dtype=X.dtype, device=X.device)

        # primal variable, size (n_samples, n_samples)
        log_P = log_Pse(C, self.eps_, self.mu_, eps_square=self.eps_square)

        optimizer = OPTIMIZERS[self.optimizer]([self.eps_, self.mu_], lr=self.lr)

        if self.tolog:
            self.log["eps"] = [self.eps_.clone().detach().cpu()]
            self.log["mu"] = [self.eps_.clone().detach().cpu()]

        if self.verbose:
            print(
                "[TorchDR] Affinity : Computing the Symmetric Entropic Affinity matrix."
            )

        one = torch.ones(n, dtype=X.dtype, device=X.device)
        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for k in pbar:
            with torch.no_grad():
                optimizer.zero_grad()
                H = entropy(log_P, log=True)

                if self.eps_square:
                    # the Jacobian must be corrected by 2 * diag(eps)
                    self.eps_.grad = (
                        2 * self.eps_.clone().detach() * (H - target_entropy)
                    )
                else:
                    self.eps_.grad = H - target_entropy

                P_sum = log_P.logsumexp(1).exp().squeeze()  # squeeze for keops
                self.mu_.grad = P_sum - one
                optimizer.step()
                if not self.eps_square:  # optimize on eps > 0
                    self.eps_.clamp_(min=0)

                log_P = log_Pse(C, self.eps_, self.mu_, eps_square=self.eps_square)

                check_NaNs(
                    [self.eps_, self.mu_],
                    msg=f"[TorchDR] Affinity (ERROR): NaN at iter {k}.",
                )

                if self.tolog:
                    self.log["eps"].append(self.eps_.clone().detach())
                    self.log["mu"].append(self.mu_.clone().detach())

                perps = (H - 1).exp()
                if self.verbose:
                    pbar.set_description(
                        f"PERPLEXITY:{float(perps.mean().item()): .2e} "
                        f"(std:{float(perps.std().item()): .2e}), "
                        f"MARGINAL:{float(P_sum.mean().item()): .2e} "
                        f"(std:{float(P_sum.std().item()): .2e})"
                    )

                if ((H - np.log(self.perplexity) - 1).abs() < self.tol).all() and (
                    (P_sum - one).abs() < self.tol
                ).all():
                    if self.verbose:
                        print(f"[TorchDR] Affinity : breaking at iter {k}.")
                    break

                if k == self.max_iter - 1 and self.verbose:
                    print(
                        "[TorchDR] Affinity (WARNING) : max iter attained, algorithm \
                            stops but may not have converged."
                    )

        self.n_iter_ = k
        self.log_affinity_matrix_ = log_P


class DoublyStochasticEntropic(LogAffinity):
    r"""
    Computes the symmetric doubly stochastic affinity matrix with controlled global
    entropy using Sinkhorn algorithm.

    Parameters
    ----------
    eps : float, optional
        Regularization parameter for the Sinkhorn algorithm.
    f : torch.Tensor of shape (n_samples), optional
        Initialization for the dual variable of the Sinkhorn algorithm (default None).
    tol : float, optional
        Precision threshold at which the algorithm stops.
    max_iter : int, optional
        Number of maximum iterations for the algorithm.
    student : bool, optional
        Whether to use a t-Student kernel instead of a Gaussian kernel.
    verbose : bool, optional
        Verbosity.
    tolog : bool, optional
        Whether to store intermediate result in a dictionary.

    Attributes
    ----------
    log_ : dictionary
        Contains the dual variables at each iteration of the optimization algorithm
        when tolog = True.
    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [3] Interpolating between Optimal Transport and MMD using Sinkhorn Divergences.
        Jean Feydy, Thibault Séjourné, François-Xavier Vialard, et al. AISTATS 2019.
    """

    def __init__(
        self,
        eps=1.0,
        f=None,
        tol=1e-5,
        max_iter=100,
        student=False,
        verbose=False,
        tolog=False,
        metric="euclidean",
        keops=True,
    ):
        self.eps = eps
        self.f = f
        self.tol = tol
        self.max_iter = max_iter
        self.student = student
        self.tolog = tolog
        self.verbose = verbose
        self.metric = metric
        self.keops = keops
        super(DoublyStochasticEntropic, self).__init__()

    def fit(self, X):
        super().fit(X)

        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        if self.student:
            C = (1 + C).log()

        if self.verbose:
            print(
                "[TorchDR] Affinity : Computing the (KL) Doubly Stochastic \
                    Affinity matrix."
            )

        n = C.shape[0]
        log_K = -C / self.eps

        # Performs warm-start if a dual variable f is provided
        f = torch.zeros(n, dtype=X.dtype, device=X.device) if self.f is None else self.f

        if self.tolog:
            self.log["f"] = [f.clone().detach().cpu()]

        # Sinkhorn iterations
        for k in range(self.max_iter):

            # well conditioned symmetric Sinkhorn iteration from Feydy et al. (2019)
            reduction = -sum_matrix_vector(log_K, f).logsumexp(0).squeeze()
            f = 0.5 * (f + reduction)

            if self.tolog:
                self.log["f"].append(f.clone().detach().cpu())

            check_NaNs(f, msg=f"[TorchDR] Affinity (ERROR): NaN at iter {k}.")

            if ((f - reduction).abs() < self.tol).all():
                if self.verbose:
                    print(f"[TorchDR] Affinity : breaking at iter {k}.")
                break

            if k == self.max_iter - 1 and self.verbose:
                print(
                    "[TorchDR] Affinity (WARNING) : max iter attained, \
                        algorithm stops but may not have converged."
                )

        self.n_iter_ = k
        self.log_affinity_matrix_ = log_Pds(log_K, f)
