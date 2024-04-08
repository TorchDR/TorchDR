# -*- coding: utf-8 -*-
"""
Entropic projections to construct (entropic) affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from tqdm import tqdm
from pykeops.torch import LazyTensor

from torchdr.utils.optim import false_position, OPTIMIZERS
from torchdr.utils.geometry import pairwise_distances
from torchdr.affinity.base import BaseAffinity, LogAffinity


class NanError(Exception):
    pass


class BadPerplexity(Exception):
    pass


def entropy(P, log=True, dim=1):
    r"""
    Computes the entropy of P along axis dim.
    Supports log domain input.
    Output of shape (n_samples, 1) to be consistent with keops.

    Parameters
    ----------
    P : tensor or lazy tensor of shape (n_samples, n_samples)
        Input matrix.
    log : bool
        If True, assumes that P is in log domain.
    dim : int or tuple
        Axis on which entropy is computed.

    Returns
    -------
    entropy : tensor of shape (n_samples, 1)
        entropy of P along axis dim.
    """
    if log:
        return -(P.exp() * (P - 1)).sum(dim).view(-1, 1)
    else:
        return -(P * (P.log() - 1)).sum(dim).view(-1, 1)


def log_Pe(C, eps):
    r"""
    Returns the log of the directed entropic affinity matrix defined in [1]
    with prescribed kernel bandwidth epsilon.
    eps has shape (n_samples, 1) to be consistent with keops.

    Parameters
    ----------
    C : tensor or lazy tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    eps : tensor of shape (n_samples, 1)
        Kernel bandwidths vector.

    Returns
    -------
    log_P : lazy tensor of shape (n_samples, n_samples)
        log of the directed affinity matrix of SNE.

    References
    ----------
    .. [1] Stochastic Neighbor Embedding.
        Geoffrey Hinton, Sam Roweis, NeurIPS 2002.
    """
    if isinstance(C, LazyTensor):
        eps = LazyTensor(eps, 0)
    log_P = -C / eps
    return log_P - log_P.logsumexp(1)[:, None]


def log_Pse(C, eps, mu, eps_square=False):
    r"""
    Returns the log of the symmetric entropic affinity matrix defined in [1]
    with given (dual) variables epsilon and mu.
    eps and mu have shape (n_samples, 1) to be consistent with keops.

    Parameters
    ----------
    C : tensor or lazy tensor of shape (n_samples, n_samples)
        Distance matrix between samples.
    eps : tensor of shape (n_samples, 1)
        Sym. entropic affinity dual variables associated with the entropy constraint.
    mu : tensor of shape (n_samples, 1)
        Sym. entropic affinity dual variables associated with the marginal constraint.
    eps_square : bool, optional
        Whether to use the square of the dual variables associated with the entropy
        constraint, by default False.

    Returns
    -------
    log_P : tensor or lazy tensor of shape (n_samples, n_samples)
        log of the symmetric entropic affinity matrix.

    References
    ----------
    .. [2] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities.
        Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    """
    if isinstance(C, LazyTensor):
        eps_t = LazyTensor(eps, 1)
        mu_t = LazyTensor(mu, 1)
        eps = LazyTensor(eps, 0)
        mu = LazyTensor(mu, 0)
    else:
        eps_t = eps.T
        mu_t = mu.T

    if eps_square:
        return (mu + mu_t - 2 * C) / (eps**2 + eps_t**2)
    else:
        return (mu + mu_t - 2 * C) / (eps + eps_t)


def log_Pds(log_K, f):
    r"""
    Returns the log of the doubly stochastic normalization of log_K (in log domain)
    given a scaling vector f which can be computed via the Sinkhorn iterations
    described in [3].
    f has shape (n_samples, 1) to be consistent with keops.

    Parameters
    ----------
    log_K : tensor or lazy tensor of shape (n_samples, n_samples)
        Log of the kernel matrix to normalize as doubly stochastic.
    f : tensor of shape (n_samples, 1)
        Scaling vector or dual variable of symmetric OT divided by entropic regularizer.

    Returns
    -------
    log_P : lazy tensor of shape (n_samples, n_samples)
        Log of the doubly stochastic affinity matrix.

    References
    ----------
    .. [3] Interpolating between Optimal Transport and MMD using Sinkhorn Divergences.
        Jean Feydy, Thibault Séjourné, François-Xavier Vialard, et al. AISTATS 2019.
    """
    if isinstance(log_K, LazyTensor):
        f_t = LazyTensor(f, 1)
        f = LazyTensor(f, 0)
    else:
        f_t = f.T

    return f + f_t + log_K


class EntropicAffinity(LogAffinity):
    r"""
    Solves the (directed) entropic affinity problem introduced in [1].
    Also corresponds to the Pe matrix in [2] (see also [3]).

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
    .. [2] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities.
        Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    .. [4] Entropic Affinities: Properties and Efficient Numerical Computation.
        Max Vladymyrov, Miguel A. Carreira-Perpinan, ICML 2013.
    .. [5] Visualizing Data using t-SNE.
        Laurens Van der Maaten, Geoffrey Hinton, JMLR 2008.
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

    def compute_log_affinity(self, X):
        r"""
        Computes the pairwise entropic affinity matrix in log space.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P : tensor or lazy tensor (if keops is True) of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        self.dtype = X.dtype
        self.device = X.device
        log_P = self._solve_dual(C)
        return log_P

    def _solve_dual(self, C):
        r"""
        Solves the problem (EA) in [1] and returns the entropic affinity matrix in
        log space (which is **not** symmetric).

        Parameters
        ----------
        C : tensor or lazy tensor (if keops) of shape (n_samples, n_samples)
            Distance matrix between the samples.

        Returns
        -------
        log_P : tensor or lazy tensor (if keops is True) of shape (n_samples, n_samples)
            Affinity matrix in log space.

        References
        ----------
        .. [2] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities, Hugues
            Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        target_entropy = np.log(self.perplexity) + 1
        n = C.shape[0]

        if not 1 < self.perplexity <= n:
            BadPerplexity(
                "The perplexity parameter must be between 1 and number of samples"
            )

        def entropy_gap(eps):  # function to find the root of
            return entropy(log_Pe(C, eps), log=True) - target_entropy

        begin, end = self.init_bounds(C)

        eps_star = false_position(
            f=entropy_gap,
            n=n,
            begin=begin,
            end=end,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            dtype=self.dtype,
            device=self.device,
        )
        log_affinity = log_Pe(C, eps_star)

        return log_affinity

    def init_bounds(self, C):
        r"""
        Computes the bounds derived in [4] for the entropic affinity root.

        Parameters
        ----------
        C : tensor or lazy tensor (if keops) of shape (n_samples, n_samples)
            Distance matrix between the samples.

        Returns
        -------
        begin : tensor of shape (n_samples, 1)
            Lower bound of the root.
        end : tensor of shape (n_samples, 1)
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
            return np.log(np.min([np.sqrt(2 * N), self.perplexity])) - 2 * (
                1 - x
            ) * np.log(N / (2 * (1 - x)))

        begin = 3 / 4
        end = 1 - 1e-6
        p1 = false_position(
            f=find_p1, n=1, begin=begin, end=end, max_iter=1000, tol=1e-6, verbose=True
        ).item()

        # retrieve greatest and smallest pairwise distances
        if isinstance(C, LazyTensor):
            dN = C.max(dim=1)
            d12 = C.Kmin(K=2, dim=1)
        else:
            dN = C.max(dim=1).values.view(-1, 1)  # for consistency with keops
            d12 = C.topk(k=2, dim=1, largest=False).values
        d1 = d12[:, 0, None]
        d2 = d12[:, 1, None]
        Delta_N = dN - d1
        Delta_2 = d2 - d1

        # compute bounds derived in [4]
        beta_L = (
            torch.stack(
                (
                    (N * np.log(N / self.perplexity)) / ((N - 1) * Delta_N),
                    torch.sqrt(np.log(N / self.perplexity) / (dN**2 - d1**2)),
                ),
                dim=0,
            )
            .max(dim=0)
            .values
        )

        beta_U = (1 / Delta_2) * np.log((N - 1) * p1 / (1 - p1))

        # convert to our notations
        begin = 1 / beta_U
        end = 1 / beta_L

        return begin, end


class L2SymmetricEntropicAffinity(BaseAffinity):
    r"""
    Computes the L2-symmetrized entropic affinity matrix of t-SNE [1].

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

    .. [5] Visualizing Data using t-SNE.
        Laurens Van der Maaten, Geoffrey Hinton, JMLR 2008.
    .. [1] Stochastic Neighbor Embedding.
        Geoffrey Hinton, Sam Roweis, NeurIPS 2002.
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

    def compute_affinity(self, X):
        r"""
        Computes the L2-symmetrized entropic affinity matrix used in t-SNE.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P : tensor or lazy tensor of shape (n_samples, n_samples)
            Affinity matrix.
        """
        EA = EntropicAffinity(
            perplexity=self.perplexity,
            metric=self.metric,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            keops=self.keops,
        )
        log_P = EA.compute_log_affinity(X)
        n = X.shape[0]
        P_sne = (log_P.exp() + log_P.exp().T) / (2 * n)
        return P_sne


class SymmetricEntropicAffinity(LogAffinity):
    r"""
    This class computes the solution to the symmetric entropic affinity
    problem described in [1], in log space.
    More precisely, it solves equation (SEA) in [1] with the dual ascent procedure.

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
    log_ : dictionary
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
    .. [2] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities,
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

    def compute_log_affinity(self, X):
        r"""
        Computes the pairwise symmetric entropic affinity matrix in log space.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P : tensor or lazy tensor (if keops is True) of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        self.dtype = X.dtype
        self.device = X.device
        log_P = self._solve_dual(C)
        return log_P

    def _solve_dual(self, C):
        r"""
        Solves the dual optimization problem (Dual-SEA) in [1] and returns the
        corresponding symmetric entropic affinty in log space.

        Parameters
        ----------
        C : lazy tensor of shape (n_samples, n_samples)
            Distance matrix between samples.

        Returns
        -------
        log_P : lazy tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.

        References
        ----------
        .. [2] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities,
            Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
        """
        n = C.shape[0]
        if not 1 < self.perplexity <= n:
            BadPerplexity(
                "The perplexity parameter must be between 1 and number of samples"
            )
        target_entropy = np.log(self.perplexity) + 1

        # dual variables, size (n_samples, 1) for keops
        eps = torch.ones((n, 1), dtype=self.dtype, device=self.device)
        mu = torch.zeros((n, 1), dtype=self.dtype, device=self.device)

        # primal variable, size (n_samples, n_samples)
        log_P = log_Pse(C, eps, mu, eps_square=self.eps_square)

        optimizer = OPTIMIZERS[self.optimizer]([eps, mu], lr=self.lr)

        if self.tolog:
            self.log_["eps"] = [eps.clone().detach().cpu()]
            self.log_["mu"] = [mu.clone().detach().cpu()]
            self.log_["loss"] = []

        if self.verbose:
            print(
                "---------- Computing the symmetric entropic affinity matrix ----------"
            )

        one = torch.ones((n, 1), dtype=self.dtype, device=self.device)
        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for k in pbar:
            with torch.no_grad():
                optimizer.zero_grad()
                H = entropy(log_P, log=True)

                if self.eps_square:
                    # the Jacobian must be corrected by 2 * diag(eps)
                    eps.grad = 2 * eps.clone().detach() * (H - target_entropy)
                else:
                    eps.grad = H - target_entropy

                P_sum = log_P.logsumexp(1).exp().view(-1, 1)  # view(-1, 1) for keops
                mu.grad = P_sum - one
                optimizer.step()
                if not self.eps_square:  # optimize on eps > 0
                    eps.clamp_(min=0)

                log_P = log_Pse(C, eps, mu, eps_square=self.eps_square)

                if torch.isnan(eps).any() or torch.isnan(mu).any():
                    raise NanError(
                        f"---------- NaN in dual variables at iteration {k}, \
                        consider decreasing the learning rate of \
                        SymmetricEntropicAffinity ----------"
                    )

                if self.tolog:
                    eps0 = eps.clone().detach()
                    mu0 = mu.clone().detach()
                    self.log_["eps"].append(eps0)
                    self.log_["mu"].append(mu0)

                perps = torch.exp(H - 1)
                if self.verbose:
                    pbar.set_description(
                        f"PERPLEXITY:{float(perps.mean().item()): .2e} "
                        f"(std:{float(perps.std().item()): .2e}), "
                        f"MARGINAL:{float(P_sum.mean().item()): .2e} "
                        f"(std:{float(P_sum.std().item()): .2e})"
                    )

                if (torch.abs(H - np.log(self.perplexity) - 1) < self.tol).all() and (
                    torch.abs(P_sum - one) < self.tol
                ).all():
                    self.log_["n_iter"] = k
                    self.n_iter_ = k
                    if self.verbose:
                        print(f"---------- Breaking at iter {k} ----------")
                    break

                if k == self.max_iter - 1 and self.verbose:
                    print(
                        "---------- Warning: max iter attained, algorithm stops but \
                            may not have converged ----------"
                    )

        self.eps_ = eps.clone().detach()
        self.mu_ = mu.clone().detach()

        return log_P


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
        self.n_iter_ = 0
        self.verbose = verbose
        self.metric = metric
        self.keops = keops
        super(DoublyStochasticEntropic, self).__init__()

    def compute_log_affinity(self, X):
        r"""
        Computes the doubly stochastic affinity matrix in log space.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P : tensor or lazy tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        self.dtype = X.dtype
        self.device = X.device

        # if student is True, considers the Student-t kernel instead of Gibbs kernel
        if self.student:
            C = (1 + C).log()
        log_P = self._solve_dual(C)
        return log_P

    def _solve_dual(self, C):
        r"""
        Performs Sinkhorn iterations in log domain to solve the entropic "self"
        (or "symmetric") OT problem with symmetric cost C and entropic regularization
        parameter eps.

        Parameters
        ----------
        C : tensor or lazy tensor of shape (n_samples, n_samples)
            Distance matrix between samples.

        Returns
        -------
        log_P : tensor or lazy tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """

        if self.verbose:
            print(
                "---------- Computing the doubly stochastic affinity matrix ----------"
            )
        n = C.shape[0]
        log_K = -C / self.eps

        # Performs warm-start if a dual variable f is provided
        f = (
            torch.zeros((n, 1), dtype=self.dtype, device=self.device)
            if self.f is None
            else self.f
        )

        if self.tolog:
            self.log_["f"] = [f.clone().detach().cpu()]

        # Sinkhorn iterations
        for k in range(self.max_iter):

            # if keops, transform f transposed into a LazyTensor for Sinkhorn update
            if isinstance(log_K, LazyTensor):
                f_t = LazyTensor(f, 1)
            else:
                f_t = f.T

            # well conditioned symmetric Sinkhorn iteration from Feydy et al. (2019)
            reduction = -(log_K + f_t).logsumexp(1).view(-1, 1)
            f = 0.5 * (f + reduction)

            if self.tolog:
                self.log_["f"].append(f.clone().detach().cpu())

            if torch.isnan(f).any():
                raise NanError(
                    f"---------- NaN in dual variables at iteration {k} \
                        ----------"
                )

            if ((f - reduction).abs() < self.tol).all():
                if self.verbose:
                    print(f"----- Breaking at iter {k} -----")
                break

            if k == self.max_iter - 1 and self.verbose:
                print(
                    "---------- Warning: max iter attained, algorithm stops but \
                            may not have converged ----------"
                )

        self.n_iter_ = k

        return log_Pds(log_K, f)
