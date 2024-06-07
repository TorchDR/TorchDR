# -*- coding: utf-8 -*-
"""
Affinity matrices with entropic constraints
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
    entropy,
    false_position,
    wrap_vectors,
    sum_matrix_vector,
    kmin,
    kmax,
    check_NaNs,
    logsumexp_red,
    batch_transpose,
    OPTIMIZERS,
)
from torchdr.affinity.base import LogAffinity


@wrap_vectors
def _log_Pe(C, eps):
    r"""
    Returns the log of the unnormalized directed entropic affinity matrix
    with prescribed kernel bandwidth epsilon.
    """
    return -C / eps


@wrap_vectors
def _log_Pse(C, eps, mu, eps_square=False):
    r"""
    Returns the log of the symmetric entropic affinity matrix
    with given (dual) variables epsilon and mu.
    """
    _eps = eps**2 if eps_square else eps
    mu_t = batch_transpose(mu)
    _eps_t = batch_transpose(_eps)
    return (mu + mu_t - 2 * C) / (_eps + _eps_t)


@wrap_vectors
def _log_Pds(log_K, dual):
    r"""
    Returns the log of the doubly stochastic normalization of log_K (in log domain)
    given a scaling vector (dual variable) which can be computed via the
    Sinkhorn iterations.
    """
    dual_t = batch_transpose(dual)
    return dual + dual_t + log_K


def _bounds_entropic_affinity(C, perplexity):
    r"""
    Computes the bounds derived in [4]_ for the entropic affinity root.

    Parameters
    ----------
    C : tensor or lazy tensor of shape (n_samples, n_samples)
        or (n_samples, k) if sparsity is used
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
    # we use the same notations as in [4] for clarity purposes
    N = C.shape[0]

    # solve a unique 1D root finding problem
    def find_p1(x):
        return np.log(np.min([np.sqrt(2 * N), perplexity])) - 2 * (1 - x) * np.log(
            N / (2 * (1 - x))
        )

    begin = 3 / 4
    end = 1 - 1e-6
    p1 = false_position(
        f=find_p1, n=1, begin=begin, end=end, max_iter=1000, tol=1e-6, verbose=False
    ).item()

    # retrieve greatest and smallest pairwise distances
    dN = kmax(C, k=1, dim=1)[0].squeeze()
    d12 = kmin(C, k=2, dim=1)[0]
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

    The entropic affinity matrix is akin to a **soft** :math:`k` **-NN affinity**, with the perplexity parameter :math:`\xi` acting as :math:`k`. Each point distributes a unit mass among its closest neighbors while minimizing a transport cost given by :math:`\mathbf{C}`.

    The entropic constraint is saturated at the optimum and governs mass spread. With small :math:`\xi`, mass concentrates on a few neighbors; with large :math:`\xi`, it spreads across more neighbors thus capturing larger scales of dependencies.

    The algorithm computes the optimal dual variable :math:`\mathbf{\varepsilon}^* \in \mathbb{R}^n_{>0}` such that

    .. math::
        \forall i, \: \mathrm{h}(\mathbf{P}^{\mathrm{e}}_{i:}) = \log (\xi) + 1 \quad \text{where} \quad \forall (i,j), \: P^{\mathrm{e}}_{ij} = \frac{\exp(- C_{ij} / \varepsilon_i^\star)}{\sum_{\ell} \exp(- C_{i\ell} / \varepsilon_i^\star)}   \:.

    :math:`\mathbf{\varepsilon}^*` is computed by performing one dimensional searches since rows of :math:`\mathbf{P}^{\mathrm{e}}` are independent subproblems.

    .. note:: Symmetric versions are also available. For the affinity matrix of t-SNE [2]_, use :class:`~torchdr.affinity.L2SymmetricEntropicAffinity`. For the affinity matrix of SNEkhorn/t-SNEkhorn [3]_, use :class:`~torchdr.affinity.SymmetricEntropicAffinity`.

    Parameters
    ----------
    perplexity : float, optional
        Perplexity parameter, related to the number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
    tol : float, optional
        Precision threshold at which the root finding algorithm stops.
    max_iter : int, optional
        Number of maximum iterations for the root finding algorithm.
    sparsity: bool, optional
        If True, keeps only the 3 * perplexity smallest element on each row of
        the ground cost matrix. Recommended if perplexity is small (<50).
    metric : str, optional
        Metric to use for computing distances (default "euclidean").
    device : str, optional
        Device to use for computation.
    keops : bool, optional
        Whether to use KeOps for computation.
    verbose : bool, optional
        Verbosity.

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
        perplexity: float = 30,
        tol: float = 1e-3,
        max_iter: int = 1000,
        sparsity: bool = None,
        metric: str = "euclidean",
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.perplexity = perplexity
        self.tol = tol
        self.max_iter = max_iter
        self._sparsity = self.perplexity < 50 if sparsity is None else sparsity

        if sparsity and self.perplexity > 100 and verbose:
            print(
                "[TorchDR] Affinity (WARNING): sparsity is not recommended "
                "for a large value of perplexity."
            )

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Solves the problem (EA) in [1]_ to compute the entropic affinity matrix
        from input data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : EntropicAffinity
            The fitted instance.
        """
        super().fit(X)

        C_full = self._ground_cost_matrix(self.data_)
        if self._sparsity and self.verbose:
            print(
                "[TorchDR] Affinity : sparsity mode enabled, computing "
                "nearest neighbors."
            )
            # when using sparsity, we construct a reduced distance matrix
            # of shape (n_samples, k) where k is 3 * perplexity.
            C_reduced, self.indices_ = kmin(C_full, k=3 * self.perplexity, dim=1)
        else:
            C_reduced = C_full

        target_entropy = np.log(self.perplexity) + 1
        n = X.shape[0]

        if not 1 < self.perplexity <= n:
            raise ValueError(
                "[TorchDR] Affinity : The perplexity parameter must be greater than "
                "1 and smaller than the number of samples "
                "(not possible if n_samples = 1)."
            )

        def entropy_gap(eps):  # function to find the root of
            log_P = _log_Pe(C_reduced, eps)
            log_P_normalized = log_P - logsumexp_red(log_P, dim=1)
            return entropy(log_P_normalized, log=True) - target_entropy

        begin, end = _bounds_entropic_affinity(C_reduced, self.perplexity)
        begin += 1e-6  # avoid numerical issues

        self.eps_ = false_position(
            f=entropy_gap,
            n=n,
            begin=begin,
            end=end,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            dtype=self.data_.dtype,
            device=self.data_.device,
        )

        log_P_final = _log_Pe(C_full, self.eps_)
        self.log_normalization = logsumexp_red(log_P_final, dim=1)
        self.log_affinity_matrix_ = log_P_final - self.log_normalization

        return self

    def get_batch(self, indices: torch.Tensor, log: bool = False):
        super().get_batch(indices)
        data_batch = self.data_[indices]
        C_batch = self._ground_cost_matrix(data_batch)
        eps_batch = self.eps_[indices]
        log_P_batch = _log_Pe(C_batch, eps_batch) - self.log_normalization[indices]

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()


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
    sparsity: bool, optional
        If True, keeps only the 3 * perplexity smallest element on each row of
        the ground cost matrix. Recommended if perplexity is small (<50).
    metric: str, optional
        Metric to use for computing distances, by default "euclidean".
    device : str, optional
        Device to use for computation.
    keops : bool, optional
        Whether to use KeOps for computation.
    verbose : bool, optional
        Verbosity.

    References
    ----------
    .. [2]  Laurens van der Maaten, Geoffrey Hinton (2008).
            Visualizing Data using t-SNE.
            The Journal of Machine Learning Research 9.11 (JMLR).
    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        tol: float = 1e-5,
        max_iter: int = 1000,
        sparsity: bool = None,
        metric: str = "euclidean",
        device: str = None,
        keops: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            perplexity=perplexity,
            tol=tol,
            max_iter=max_iter,
            sparsity=sparsity,
            metric=metric,
            device=device,
            keops=keops,
            verbose=verbose,
        )

    def fit(self, X: torch.Tensor | np.ndarray):
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
        self.N = self.data_.shape[0]
        self.affinity_matrix_ = (log_P.exp() + log_P.exp().T) / (2 * self.N)
        self.log_affinity_matrix_ = self.affinity_matrix_.log()

    def get_batch(self, indices: torch.Tensor):
        P_batch = super().get_batch(indices, log=False)
        return (P_batch + batch_transpose(P_batch)) / (2 * self.N)


class SymmetricEntropicAffinity(LogAffinity):
    r"""
    Computes the solution :math:`\mathbf{P}^{\mathrm{se}}` to the symmetric entropic
    affinity problem (SEA) described in [3]_ with the dual ascent procedure.
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

    The algorithm computes the optimal dual variables :math:`\mathbf{\mu}^\star \in \mathbb{R}^n` and :math:`\mathbf{\varepsilon}^\star \in \mathbb{R}^n_{>0}` using dual ascent. The affinity matrix is then given by

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
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-5.
    max_iter : int, optional
        Number of maximum iterations for the algorithm, by default 500.
    optimizer : {'SGD', 'Adam', 'NAdam', 'LBFGS}, optional
        Which pytorch optimizer to use (default 'Adam').
    tolog : bool, optional
        Whether to store intermediate result in a dictionary (default False).
    metric : str, optional
        Metric to use for computing distances, by default "euclidean".
    device : str, optional
        Device to use for computation.
    keops : bool, optional
        Whether to use KeOps for computation.
    verbose : bool, optional
        Verbosity (default True).

    References
    ----------
    .. [3] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities,
        Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty, NeurIPS 2023.
    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        lr: float = 1e0,
        eps_square: bool = True,
        tol: float = 1e-3,
        max_iter: int = 500,
        optimizer: str = "Adam",
        metric: str = "euclidean",
        device: str = None,
        keops: bool = False,
        verbose: bool = True,
        tolog: bool = False,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.perplexity = perplexity
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.tolog = tolog
        self.eps_square = eps_square

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Solves the problem (SEA) in [3]_ to compute the symmetric entropic affinity
        matrix from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : SymmetricEntropicAffinity
            The fitted instance.
        """
        self.log_ = {}
        if self.verbose:
            print(
                "[TorchDR] Affinity : Computing the Symmetric Entropic Affinity matrix."
            )

        super().fit(X)

        C = self._ground_cost_matrix(self.data_)

        n = C.shape[0]
        if not 1 < self.perplexity <= n:
            raise ValueError(
                "[TorchDR] Affinity : The perplexity parameter must be greater than "
                "1 and smaller than the number of samples "
                "(not possible if n_samples = 1)."
            )

        target_entropy = np.log(self.perplexity) + 1
        one = torch.ones(n, dtype=self.data_.dtype, device=self.data_.device)

        # dual variables, size (n_samples)
        self.eps_ = torch.ones(n, dtype=self.data_.dtype, device=self.data_.device)
        self.mu_ = torch.ones(n, dtype=self.data_.dtype, device=self.data_.device)

        if self.optimizer == "LBFGS":

            self.eps_.requires_grad = True
            self.mu_.requires_grad = True

            optimizer = torch.optim.LBFGS(
                [self.eps_, self.mu_],
                lr=self.lr,
                max_iter=self.max_iter,
                tolerance_grad=self.tol,
                line_search_fn="strong_wolfe",
            )

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                _eps = self.eps_**2 if self.eps_square else self.eps_
                log_P = _log_Pse(C, _eps, self.mu_, eps_square=False)
                H = entropy(log_P, log=True, dim=1)
                loss = (  # Negative Lagrangian loss
                    -(log_P.exp() * C).sum(0).sum()
                    - torch.inner(_eps, target_entropy - H)
                    + torch.inner(self.mu_, log_P.logsumexp(1).squeeze().expm1())
                )
                if loss.requires_grad:
                    loss.backward()
                return loss

            optimizer.step(closure)

            check_NaNs(
                [self.eps_, self.mu_],
                msg=(
                    "[TorchDR] Affinity (ERROR): NaN in dual variables, "
                    "consider decreasing the learning rate."
                ),
            )

            if self.tolog:
                self.log_["eps"] = [self.eps_.clone().detach().cpu()]
                self.log_["mu"] = [self.eps_.clone().detach().cpu()]

            self.log_affinity_matrix_ = _log_Pse(
                C, self.eps_, self.mu_, eps_square=self.eps_square
            )

        else:  # other optimizers including SGD and Adam

            optimizer = OPTIMIZERS[self.optimizer]([self.eps_, self.mu_], lr=self.lr)

            if self.tolog:
                self.log_["eps"] = [self.eps_.clone().detach().cpu()]
                self.log_["mu"] = [self.eps_.clone().detach().cpu()]

            pbar = tqdm(range(self.max_iter), disable=not self.verbose)
            for k in pbar:
                with torch.no_grad():
                    optimizer.zero_grad()

                    log_P = _log_Pse(C, self.eps_, self.mu_, eps_square=self.eps_square)
                    H = entropy(log_P, log=True, dim=1)
                    P_sum = log_P.logsumexp(1).exp().squeeze()  # squeeze for keops

                    grad_eps = H - target_entropy
                    if self.eps_square:
                        # the Jacobian must be corrected by 2 * diag(eps)
                        grad_eps = 2 * self.eps_.clone().detach() * grad_eps
                    grad_mu = P_sum - one

                    self.eps_.grad = grad_eps
                    self.mu_.grad = grad_mu
                    optimizer.step()

                    if not self.eps_square:  # optimize on eps > 0
                        self.eps_.clamp_(min=0)

                    check_NaNs(
                        [self.eps_, self.mu_],
                        msg=(
                            f"[TorchDR] Affinity (ERROR): NaN at iter {k}, "
                            "consider decreasing the learning rate."
                        ),
                    )

                    if self.tolog:
                        self.log_["eps"].append(self.eps_.clone().detach())
                        self.log_["mu"].append(self.mu_.clone().detach())

                    perps = (H - 1).exp()
                    if self.verbose:
                        pbar.set_description(
                            f"PERPLEXITY:{float(perps.mean().item()): .2e} "
                            f"(std:{float(perps.std().item()): .2e}), "
                            f"MARGINAL:{float(P_sum.mean().item()): .2e} "
                            f"(std:{float(P_sum.std().item()): .2e})"
                        )

                    if (
                        torch.norm(grad_eps) < self.tol
                        and torch.norm(grad_mu) < self.tol
                    ):
                        if self.verbose:
                            print(
                                "[TorchDR] Affinity : convergence reached "
                                f"at iter {k}."
                            )
                        break

                    if k == self.max_iter - 1 and self.verbose:
                        print(
                            "[TorchDR] Affinity (WARNING) : max iter attained, "
                            "algorithm stops but may not have converged."
                        )

            self.n_iter_ = k
            self.log_affinity_matrix_ = log_P

    def get_batch(self, indices: torch.Tensor, log: bool = False):
        super().get_batch(indices)
        data_batch = self.data_[indices]
        C_batch = self._ground_cost_matrix(data_batch)
        eps_batch = self.eps_[indices]
        mu_batch = self.mu_[indices]
        log_P_batch = _log_Pse(C_batch, eps_batch, mu_batch, eps_square=self.eps_square)

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()


class DoublyStochasticEntropic(LogAffinity):
    r"""
    Computes the symmetric doubly stochastic affinity matrix :math:`\mathbf{P}^{\mathrm{ds}}` with controlled global entropy using the symmetric Sinkhorn algorithm [5]_.
    Consists in solving the following symmetric entropic optimal transport problem [6]_:

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

    The algorithm computes the optimal dual variable
    :math:`\mathbf{f}^\star \in \mathbb{R}^n` such that

    .. math::
        \mathbf{P}^{\mathrm{ds}} \mathbf{1} = \mathbf{1} \quad \text{where} \quad \forall (i,j), \: P^{\mathrm{ds}}_{ij} = \exp(f^\star_i + f^\star_j - C_{ij} / \varepsilon) \:.

    :math:`\mathbf{f}^\star` is computed by performing dual ascent via the Sinkhorn fixed-point iteration (equation in 25 in [7]_).

    **Bregman projection.** Another way to write this problem is to consider the
    KL projection of the Gibbs kernel onto the set of doubly stochastic matrices:

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
    base_kernel : {"gaussian", "student"}, optional
        Which base kernel to normalize as doubly stochastic.
    tolog : bool, optional
        Whether to store intermediate result in a dictionary.
    metric : str, optional
        Metric to use for computing distances (default "euclidean").
    device : str, optional
        Device to use for computation.
    keops : bool, optional
        Whether to use KeOps for computation.
    verbose : bool, optional
        Verbosity.

    References
    ----------
    .. [5]  Richard Sinkhorn, Paul Knopp (1967).
            Concerning nonnegative matrices and doubly stochastic matrices.
            Pacific Journal of Mathematics, 21(2), 343-348.

    .. [6]  Marco Cuturi (2013).
            Sinkhorn Distances: Lightspeed Computation of Optimal Transport.
            Advances in Neural Information Processing Systems 26 (NeurIPS).

    .. [7]  Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari,
            Alain Trouvé, Gabriel Peyré (2019).
            Interpolating between Optimal Transport and MMD using Sinkhorn Divergences.
            International Conference on Artificial Intelligence and Statistics
            (AISTATS).
    """  # noqa: E501

    def __init__(
        self,
        eps: float = 1.0,
        init_dual: torch.Tensor = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        base_kernel: str = "gaussian",
        tolog: bool = False,
        metric: str = "euclidean",
        device: str = None,
        keops: bool = False,
        verbose: bool = False,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.eps = eps
        self.init_dual = init_dual
        self.tol = tol
        self.max_iter = max_iter
        self.base_kernel = base_kernel
        self.tolog = tolog

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""Computes the entropic doubly stochastic affinity matrix from input data X.

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

        C = self._ground_cost_matrix(self.data_)
        if self.base_kernel == "student":
            C = (1 + C).log()

        if self.verbose:
            print(
                "[TorchDR] Affinity : Computing the (KL) Doubly Stochastic "
                "Affinity matrix."
            )

        n = C.shape[0]
        log_K = -C / self.eps

        # Performs warm-start if a dual variable f is provided
        self.dual_ = (
            torch.zeros(n, dtype=self.data_.dtype, device=self.data_.device)
            if self.init_dual is None
            else self.init_dual
        )

        if self.tolog:
            self.log["dual"] = [self.dual_.clone().detach().cpu()]

        # Sinkhorn iterations
        for k in range(self.max_iter):

            # well conditioned symmetric Sinkhorn iteration from Feydy et al. (2019)
            reduction = -sum_matrix_vector(log_K, self.dual_).logsumexp(0).squeeze()
            self.dual_ = 0.5 * (self.dual_ + reduction)

            if self.tolog:
                self.log["dual"].append(self.dual_.clone().detach().cpu())

            check_NaNs(self.dual_, msg=f"[TorchDR] Affinity (ERROR): NaN at iter {k}.")

            if torch.norm(self.dual_ - reduction) < self.tol:
                if self.verbose:
                    print(f"[TorchDR] Affinity : convergence reached at iter {k}.")
                break

            if k == self.max_iter - 1 and self.verbose:
                print(
                    "[TorchDR] Affinity (WARNING) : max iter attained, algorithm "
                    "stops but may not have converged."
                )

        self.n_iter_ = k
        self.log_affinity_matrix_ = _log_Pds(log_K, self.dual_)

        return self
