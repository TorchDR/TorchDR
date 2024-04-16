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

from torchdr.utils import (
    false_position,
    pairwise_distances,
    wrap_vectors,
    sum_matrix_vector,
    kmin,
    kmax,
    check_NaNs,
    OPTIMIZERS,
)
from torchdr.affinity._base import LogAffinity


def entropy(P, log=True, dim=1):
    r"""
    Computes the entropy of P along axis dim.
    Supports log domain input.
    """
    if log:
        return -(P.exp() * (P - 1)).sum(dim).squeeze()
    else:
        return -(P * (P.log() - 1)).sum(dim).squeeze()


@wrap_vectors
def log_Pe(C, eps):
    r"""
    Returns the log of the directed entropic affinity matrix
    with prescribed kernel bandwidth epsilon.
    """
    log_P = -C / eps
    return log_P - log_P.logsumexp(1)[:, None]


@wrap_vectors
def log_Pse(C, eps, mu, eps_square=False):
    r"""
    Returns the log of the symmetric entropic affinity matrix
    with given (dual) variables epsilon and mu.
    """
    if eps_square:
        eps = eps**2

    return (mu + mu.T - 2 * C) / (eps + eps.T)


@wrap_vectors
def log_Pds(log_K, f):
    r"""
    Returns the log of the doubly stochastic normalization of log_K (in log domain)
    given a scaling vector f which can be computed via the Sinkhorn iterations.
    """
    return f + f.T + log_K


def bounds_entropic_affinity(C, perplexity):
    r"""
    Computes the bounds derived in [4]_ for the entropic affinity root.

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
    .. [4]  Max Vladymyrov, Miguel A. Carreira-Perpinan (2013).
            Entropic Affinities: Properties and Efficient Numerical Computation.
            International Conference on Machine Learning (ICML).
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
    Solves the directed entropic affinity problem introduced in [1]_.
    Corresponds to the matrix :math:`\mathbf{P}^{\mathrm{e}}` in [3]_, solving the convex optimization problem

    .. math::
        \mathbf{P}^{\mathrm{e}} \in \mathop{\arg\min}_{\mathbf{P} \in \mathbb{R}_+^{n \times n}} \: &\langle \mathbf{C}, \mathbf{P} \rangle \\
        \text{s.t.} \quad  &\mathbf{P} \mathbf{1} = \mathbf{1} \\
                            &\forall i, \: \mathrm{h}(\mathbf{P}_{i:}) \geq \log (\xi) + 1 \:.

    where :

    - :math:`\mathbf{C}`: symmetric pairwise distance matrix between the samples.
    - :math:`\xi`: perplexity parameter.
    - :math:`\mathrm{h}`: (row-wise) Shannon entropy such that :math:`\mathrm{h}(\mathbf{p}) = - \sum_{i} p_{i} (\log p_{i} - 1)`.
    - :math:`\mathbf{1} := (1,...,1)^\top`: all-ones vector.

    The entropic affinity matrix is akin to a **soft** :math:`k` **-NN affinity**, with the perplexity parameter :math:`\xi` acting as :math:`k`. Each point distributes a unit mass among its closest neighbors to minimize a transport cost given by :math:`\mathbf{C}`.

    The entropic constraint is saturated at the optimum and governs mass spread. With small :math:`\xi`, mass concentrates on a few neighbors; with large :math:`\xi`, it spreads across more neighbors thus capturing larger scales of dependencies.

    The algorithm computes the optimal dual variable :math:`\mathbf{\varepsilon}^* \in \mathbb{R}^n_{>0}` such that

    .. math::
        \forall i, \: \mathrm{h}(\mathbf{P}^{\mathrm{e}}_{i:}) = \log (\xi) + 1 \quad \text{where} \quad \forall (i,j), \: P^{\mathrm{e}}_{ij} = \frac{\exp(- C_{ij} / \varepsilon_i^\star)}{\sum_{\ell} \exp(- C_{i\ell} / \varepsilon_i^\star)}   \:.

    :math:`\mathbf{\varepsilon}^*` is computed by performing one dimensional searches since rows of :math:`\mathbf{P}^{\mathrm{e}}` are independent subproblems.

    .. note:: Symmetric versions are also available. For the affinity matrix of t-SNE [2]_, use :class:`~torchdr.affinity.L2SymmetricEntropicAffinity`. For the affinity matrix of SNEkhorn/t-SNEkhorn [3]_, use :class:`~torchdr.affinity.SymmetricEntropicAffinity`.

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
        Whether to use KeOps for computation.

    Attributes
    ----------
    log : dictionary
        Contains the dual variable at each iteration of the
        optimization algorithm when tolog = True.
    n_iter_ : int
        Number of iterations run.
    eps_ : torch.Tensor of shape (n_samples)
        Dual variable associated with the entropy constraint.

    References
    ----------
    .. [1]  Geoffrey Hinton, Sam Roweis (2002).
            Stochastic Neighbor Embedding.
            Advances in neural information processing systems 15 (NeurIPS).

    .. [2]  Laurens van der Maaten, Geoffrey Hinton (2008).
            Visualizing Data using t-SNE.
            The Journal of Machine Learning Research 9.11 (JMLR).

    .. [3]  Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty (2023).
            SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities.
            Advances in Neural Information Processing Systems 36 (NeurIPS).
    """  # noqa: E501

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
        super().__init__()

    def fit(self, X):
        r"""
        Solves the problem (EA) in [1]_ to compute the entropic affinity matrix from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : EntropicAffinity
            The fitted instance.
        """  # noqa: E501
        super().fit(X)

        C = pairwise_distances(X, metric=self.metric, keops=self.keops)

        target_entropy = np.log(self.perplexity) + 1
        n = X.shape[0]

        if not 1 < self.perplexity <= n:
            raise ValueError(
                "[TorchDR] Affinity : The perplexity parameter must be between "
                "2 and number of samples."
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

        return self


class L2SymmetricEntropicAffinity(EntropicAffinity):
    r"""
    Computes the L2-symmetrized entropic affinity matrix :math:`\overline{\mathbf{P}^{\mathrm{e}}}` of t-SNE [2]_.

    From the :class:`~torchdr.affinity.EntropicAffinity` matrix :math:`\mathbf{P}^{\mathrm{e}}`, it is computed as

    .. math::
        \overline{\mathbf{P}^{\mathrm{e}}} = \frac{\mathbf{P}^{\mathrm{e}} + (\mathbf{P}^{\mathrm{e}})^\top}{2} \:.

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
        Whether to use KeOps for computation.

    Attributes
    ----------
    log : dictionary
        Contains the dual variable at each iteration of the
        optimization algorithm when tolog = True.
    n_iter_ : int
        Number of iterations run.
    eps_ : torch.Tensor of shape (n_samples)
        Dual variable associated with the entropy constraint.

    References
    ----------
    .. [2] Visualizing Data using t-SNE.
        Laurens Van der Maaten, Geoffrey Hinton, JMLR 2008.
    """  # noqa: E501

    def __init__(
        self,
        perplexity,
        metric="euclidean",
        tol=1e-5,
        max_iter=1000,
        verbose=True,
        keops=True,
    ):
        super().__init__(
            perplexity=perplexity,
            metric=metric,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            keops=keops,
        )

    def fit(self, X):
        r"""
        Computes the l2-symmetric entropic affinity matrix from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : L2SymmetricEntropicAffinity
            The fitted instance.
        """
        super().fit(X)
        log_P = self.log_affinity_matrix_
        self.affinity_matrix_ = (log_P.exp() + log_P.exp().T) / 2
        self.log_affinity_matrix_ = self.affinity_matrix_.log()


class SymmetricEntropicAffinity(LogAffinity):
    r"""
    Computes the solution :math:`\mathbf{P}^{\mathrm{se}}` to the symmetric entropic affinity problem (SEA) described in [3]_ with the dual ascent procedure.
    It amounts to the following convex optimization problem:

    .. math::
        \mathbf{P}^{\mathrm{se}} \in \mathop{\arg\min}_{\mathbf{P} \in \mathbb{R}_+^{n \times n}} \: &\langle \mathbf{C}, \mathbf{P} \rangle \\
        \text{s.t.} \quad  &\mathbf{P} \mathbf{1} = \mathbf{1} \\
                            &\forall i, \: \mathrm{h}(\mathbf{P}_{i:}) \geq \log (\xi) + 1 \\
                            &\mathbf{P} = \mathbf{P}^\top \:.

    where :

    - :math:`\mathbf{C}`: symmetric pairwise distance matrix between the samples.
    - :math:`\xi`: perplexity parameter.
    - :math:`\mathrm{h}`: (row-wise) Shannon entropy such that :math:`\mathrm{h}(\mathbf{p}) = - \sum_{i} p_{i} (\log p_{i} - 1)`.
    - :math:`\mathbf{1} := (1,...,1)^\top`: all-ones vector.

    It is a symmetric version of :class:`~torchdr.affinity.EntropicAffinity`, where we simply added symmetry as a constraint in the optimization problem.

    The algorithm computes the optimal dual variable :math:`\mathbf{\mu}^\star \in \mathbb{R}^n` and :math:`\mathbf{\varepsilon}^\star \in \mathbb{R}^n_{>0}` using dual ascent. The affinity matrix is then given by

    .. math::
        \forall (i,j), \: P^{\mathrm{se}}_{ij} = \exp \left( \frac{\mu^\star_{i} + \mu^\star_j - 2 C_{ij}}{\varepsilon^\star_i + \varepsilon^\star_j} \right) \:.

    .. note:: Unlike :class:`~torchdr.affinity.L2SymmetricEntropicAffinity` that is used in t-SNE, :class:`~torchdr.affinity.SymmetricEntropicAffinity` enables to control the entropy and mass of each row/column of the affinity matrix.

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
        Whether to use KeOps for computation.

    Attributes
    ----------
    log : dictionary
        Contains the dual variables at each iteration of the
        optimization algorithm when tolog = True.
    n_iter_ : int
        Number of iterations run.
    eps_ : torch.Tensor of shape (n_samples)
        Dual variable associated with the entropy constraint.
    mu_ : torch.Tensor of shape (n_samples)
        Dual variable associated with the marginal constraint.

    References
    ----------
    .. [3] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities,
        Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    """  # noqa: E501

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
        super().__init__()

    def fit(self, X):
        r"""
        Solves the problem (SEA) in [3]_ to compute the symmetric entropic affinity matrix from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : SymmetricEntropicAffinity
            The fitted instance.
        """  # noqa: E501
        super().fit(X)

        C = pairwise_distances(X, metric=self.metric, keops=self.keops)

        n = X.shape[0]
        if not 1 < self.perplexity <= n:
            raise ValueError(
                "[TorchDR] Affinity : The perplexity parameter must be between "
                "1 and number of samples."
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
                H = entropy(log_P, log=True, dim=1)

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
                    msg=(
                        f"[TorchDR] Affinity (ERROR): NaN at iter {k}, "
                        "consider decreasing the learning rate."
                    ),
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
                        "[TorchDR] Affinity (WARNING) : max iter attained, algorithm "
                        "stops but may not have converged."
                    )

        self.n_iter_ = k
        self.log_affinity_matrix_ = log_P


class DoublyStochasticEntropic(LogAffinity):
    r"""
    Computes the symmetric doubly stochastic affinity matrix :math:`\mathbf{P}^{\mathrm{ds}}` with controlled global entropy using the symmetric Sinkhorn algorithm [5].
    Consists in solving the following convex optimization problem:

    .. math::
        \mathbf{P}^{\mathrm{ds}} \in \mathop{\arg\min}_{\mathbf{P} \in \mathbb{R}_+^{n \times n}} \: &\langle \mathbf{C},
        \mathbf{P} \rangle + \varepsilon \mathrm{H}(\mathbf{P}) \\
        \text{s.t.} \quad  &\mathbf{P} \mathbf{1} = \mathbf{1} \\
                            &\mathbf{P} = \mathbf{P}^\top

    where :

    - :math:`\mathbf{C}`: symmetric pairwise distance matrix between the samples.
    - :math:`\varepsilon`: entropic regularization parameter.
    - :math:`\mathrm{H}`: (global) Shannon entropy such that :math:`\mathrm{H}(\mathbf{P}) := - \sum_{ij} P_{ij} (\log P_{ij} - 1)`.
    - :math:`\mathbf{1} := (1,...,1)^\top`: all-ones vector.

    The algorithm computes the optimal dual variable :math:`\mathbf{f}^\star \in \mathbb{R}^n` such that

    .. math::
        \mathbf{P}^{\mathrm{ds}} \mathbf{1} = \mathbf{1} \quad \text{where} \quad \forall (i,j), \: P^{\mathrm{ds}}_{ij} = \exp(f^\star_i + f^\star_j - C_{ij} / \varepsilon) \:.

    :math:`\mathbf{f}^\star` is computed by performing dual ascent via the Sinkhorn fixed-point iteration.

    **Bregman projection.** Another way to write this problem is to consider the KL projection of the Gibbs kernel onto the set of doubly stochastic matrices:

    .. math::
        \mathbf{P}^{\mathrm{ds}} = \mathrm{Proj}_{\mathcal{DS}}^{\mathrm{KL}}(\mathbf{K}_\varepsilon) := \mathop{\arg\min}_{\mathbf{P} \in \mathcal{DS}} \: \mathrm{KL}(\mathbf{P} \| \mathbf{K}_\varepsilon) \:.

    where :

    - :math:`\mathbf{K}_\varepsilon := \exp(-\mathbf{C} / \varepsilon)`: Gibbs kernel.
    - :math:`\mathrm{KL}(\mathbf{P} \| \mathbf{Q}) := \sum_{ij} P_{ij} (\log (Q_{ij} / P_{ij}) - 1) + Q_{ij}`: Kullback Leibler divergence between :math:`\mathbf{P}` and :math:`\mathbf{Q}`.
    - :math:`\mathcal{DS} := \left\{ \mathbf{P} \in \mathbb{R}_+^{n \times n}: \: \mathbf{P} = \mathbf{P}^\top \:,\: \mathbf{P} \mathbf{1} = \mathbf{1} \right\}`: set of symmetric doubly stochastic matrices.

    Parameters
    ----------
    eps : float, optional
        Regularization parameter for the Sinkhorn algorithm.
    f : tensor of shape (n_samples), optional
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
    log : dictionary
        Contains the dual variable at each iteration of the
        optimization algorithm when tolog = True.
    n_iter_ : int
        Number of iterations run.
    f_ : torch.Tensor of shape (n_samples)
        Dual variable associated with the marginal constraint.

    References
    ----------
    .. [5]  Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari,
            Alain Trouvé, Gabriel Peyré (2019).
            Interpolating between Optimal Transport and MMD using Sinkhorn Divergences.
            International Conference on Artificial Intelligence and Statistics
            (AISTATS).
    """  # noqa: E501

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
        super().__init__()

    def fit(self, X):
        """Computes the entropic doubly stochastic affinity matrix from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : DoublyStochasticEntropic
            The fitted instance.
        """
        super().fit(X)

        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        if self.student:
            C = (1 + C).log()

        if self.verbose:
            print(
                "[TorchDR] Affinity : Computing the (KL) Doubly Stochastic "
                "Affinity matrix."
            )

        n = C.shape[0]
        log_K = -C / self.eps

        # Performs warm-start if a dual variable f is provided
        self.f_ = (
            torch.zeros(n, dtype=X.dtype, device=X.device) if self.f is None else self.f
        )

        if self.tolog:
            self.log["f"] = [self.f_.clone().detach().cpu()]

        # Sinkhorn iterations
        for k in range(self.max_iter):

            # well conditioned symmetric Sinkhorn iteration from Feydy et al. (2019)
            reduction = -sum_matrix_vector(log_K, self.f_).logsumexp(0).squeeze()
            self.f_ = 0.5 * (self.f_ + reduction)

            if self.tolog:
                self.log["f"].append(self.f_.clone().detach().cpu())

            check_NaNs(self.f_, msg=f"[TorchDR] Affinity (ERROR): NaN at iter {k}.")

            if ((self.f_ - reduction).abs() < self.tol).all():
                if self.verbose:
                    print(f"[TorchDR] Affinity : breaking at iter {k}.")
                break

            if k == self.max_iter - 1 and self.verbose:
                print(
                    "[TorchDR] Affinity (WARNING) : max iter attained, algorithm "
                    "stops but may not have converged."
                )

        self.n_iter_ = k
        self.log_affinity_matrix_ = log_Pds(log_K, self.f_)

        return self
