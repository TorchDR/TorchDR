"""Affinity matrices with entropic constraints."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause License

import contextlib
import math
import warnings
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from torchdr.affinity.base import LogAffinity, SparseLogAffinity
from torchdr.utils import (
    OPTIMIZERS,
    batch_transpose,
    check_NaNs,
    entropy,
    false_position,
    kmax,
    kmin,
    logsumexp_red,
    sum_matrix_vector,
    wrap_vectors,
)


@wrap_vectors
def _log_Pe(C, eps):
    r"""Return the log of the unnormalized directed entropic affinity.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        or shape (n_batch, batch_size, batch_size)
        Pairwise distance matrix.
    eps : torch.Tensor of shape (n) or (n_batch, batch_size)
        Dual variable of the entropic constraint.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        or shape (n_batch, batch_size, batch_size)
        The log of the unnormalized affinity matrix.
    """
    return -C / eps


@wrap_vectors
def _log_Pse(C, eps, mu, eps_square=False):
    r"""Return the log of the symmetric entropic affinity matrix.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        or shape (n_batch, batch_size, batch_size)
        Pairwise distance matrix.
    eps : torch.Tensor of shape (n) or (n_batch, batch_size)
        Dual variable of the entropic constraint.
    mu : torch.Tensor of shape (n) or (n_batch, batch_size)
        Dual variable of the normalization constraint.
    eps_square : bool, optional
        Whether to optimize on the square of the dual variables.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        or shape (n_batch, batch_size, batch_size)
        The log of the unnormalized affinity matrix.
    """
    _eps = eps**2 if eps_square else eps
    mu_t = batch_transpose(mu)
    _eps_t = batch_transpose(_eps)
    return (mu + mu_t - 2 * C) / (_eps + _eps_t)


@wrap_vectors
def _log_Pds(log_K, dual):
    r"""Return the log of the doubly stochastic normalization of log_K (in log domain).

    Parameters
    ----------
    log_K : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        or shape (n_batch, batch_size, batch_size)
        Log of the base kernel.
    dual : torch.Tensor of shape (n) or (n_batch, batch_size)
        Dual variable of the normalization constraint.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        or shape (n_batch, batch_size, batch_size)
        The log of the doubly stochastic normalization of log_K.
    """
    dual_t = batch_transpose(dual)
    return dual + dual_t + log_K


def _bounds_entropic_affinity(C, perplexity):
    r"""Compute the entropic affinity bounds derived in :cite:`vladymyrov2013entropic`.

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


def _check_perplexity(perplexity, n, verbose=True):
    r"""Check the perplexity parameter and return a valid value."""
    if n <= 1:
        raise ValueError(
            "[TorchDR] ERROR Affinity: Input has less than one sample : "
            f"n_samples = {n}."
        )

    if perplexity >= n or perplexity <= 1:
        new_value = 50
        if verbose:
            warnings.warn(
                "[TorchDR] WARNING Affinity: The perplexity parameter must be "
                "greater than 1 and smaller than the number of samples "
                f"(here n = {n}). Got perplexity = {perplexity}. "
                "Setting perplexity to {50}."
            )
        return new_value
    else:
        return perplexity


class EntropicAffinity(SparseLogAffinity):
    r"""Solve the directed entropic affinity problem introduced in :cite:`hinton2002stochastic`.

    The algorithm computes the optimal dual variable
    :math:`\mathbf{\varepsilon}^* \in \mathbb{R}^n_{>0}` such that

    .. math::
        \forall i, \: \mathrm{h}(\mathbf{P}^{\mathrm{e}}_{i:}) = \log (\xi) + 1 \quad \text{where} \quad \forall (i,j), \: P^{\mathrm{e}}_{ij} = \frac{\exp(- C_{ij} / \varepsilon_i^\star)}{\sum_{\ell} \exp(- C_{i\ell} / \varepsilon_i^\star)}   \:.

    where :

    - :math:`\mathbf{C}`: symmetric pairwise distance matrix between the samples.
    - :math:`\xi`: perplexity parameter.
    - :math:`\mathrm{h}`: (row-wise) Shannon entropy such that :math:`\mathrm{h}(\mathbf{p}) = - \sum_{i} p_{i} (\log p_{i} - 1)`.

    :math:`\mathbf{\varepsilon}^*` is computed by performing one dimensional searches
    since rows of :math:`\mathbf{P}^{\mathrm{e}}` are independent subproblems.

    **Convex problem.** Corresponds to the matrix :math:`\mathbf{P}^{\mathrm{e}}`
    in :cite:`van2024snekhorn`, solving the convex optimization problem

    .. math::
        \mathbf{P}^{\mathrm{e}} \in \mathop{\arg\min}_{\mathbf{P} \in \mathbb{R}_+^{n \times n}} \: &\langle \mathbf{C}, \mathbf{P} \rangle \\
        \text{s.t.} \quad  &\mathbf{P} \mathbf{1} = \mathbf{1} \\
                            &\forall i, \: \mathrm{h}(\mathbf{P}_{i:}) \geq \log (\xi) + 1 \:.

    where :math:`\mathbf{1} := (1,...,1)^\top`: is the all-ones vector.

    The entropic affinity matrix is akin to a **soft** :math:`k` **-NN affinity**,
    with the perplexity parameter :math:`\xi` acting as :math:`k`.
    Each point distributes a unit mass among its closest neighbors while minimizing
    a transport cost given by :math:`\mathbf{C}`.

    The entropic constraint is saturated at the optimum and governs mass spread.
    With small :math:`\xi`, mass concentrates on a few neighbors;
    with large :math:`\xi`, it spreads across more neighbors thus capturing
    larger scales of dependencies.

    .. note:: A symmetric version is also available at
        :class:`~torchdr.SymmetricEntropicAffinity`. It is the affinity matrix
        used in :class:`~SNEkhorn`/ :class:`~TSNEkhorn` :cite:`van2024snekhorn`. In TSNE :cite:`van2008visualizing`,
        the entropic affinity is simply averaged with its transpose.

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
        the ground cost matrix. Recommended if perplexity is not too big.
        Default is True.
    metric : str, optional
        Metric to use for computing distances (default "sqeuclidean").
    zero_diag : bool, optional
        Whether to set the diagonal of the distance matrix to 0.
    device : str, optional
        Device to use for computation.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        tol: float = 1e-3,
        max_iter: int = 1000,
        sparsity: bool = "True",
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
    ):
        self.perplexity = perplexity
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

    def _compute_sparse_log_affinity(self, X: torch.Tensor):
        r"""Solve the entropic affinity problem by :cite:`hinton2002stochastic`.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_samples, n_samples)
            Log of the entropic affinity matrix.
        indices : torch.Tensor or None
            Indices of the nearest neighbors if sparsity is used.
        """
        if self.verbose:
            print("[TorchDR] Affinity : computing the Entropic Affinity matrix.")

        n_samples_in = X.shape[0]
        perplexity = _check_perplexity(self.perplexity, n_samples_in, self.verbose)
        target_entropy = np.log(perplexity) + 1

        k = 3 * perplexity
        if self.sparsity:
            if self.verbose:
                print(
                    f"[TorchDR] Affinity : sparsity mode enabled, computing {k} "
                    "nearest neighbors. If this step is too slow, consider "
                    "reducing the dimensionality of the data or disabling sparsity."
                )
            # when using sparsity, we construct a reduced distance matrix
            # of shape (n_samples, k)
            C_, indices = self._distance_matrix(X, k=k)
        else:
            C_, indices = self._distance_matrix(X)

        def entropy_gap(eps):  # function to find the root of
            log_P = _log_Pe(C_, eps)
            log_P_normalized = log_P - logsumexp_red(log_P, dim=1)
            return entropy(log_P_normalized, log=True) - target_entropy

        begin, end = _bounds_entropic_affinity(C_, perplexity)
        begin += 1e-6  # avoid numerical issues

        self.eps_ = false_position(
            f=entropy_gap,
            n=n_samples_in,
            begin=begin,
            end=end,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            dtype=X.dtype,
            device=X.device,
        )

        log_P_final = _log_Pe(C_, self.eps_)
        self.log_normalization_ = logsumexp_red(log_P_final, dim=1)
        log_affinity_matrix = log_P_final - self.log_normalization_

        log_affinity_matrix -= math.log(n_samples_in)

        return log_affinity_matrix, indices


class SymmetricEntropicAffinity(LogAffinity):
    r"""Compute the symmetric entropic affinity (SEA) introduced in :cite:`van2024snekhorn`.

    Compute the solution :math:`\mathbf{P}^{\mathrm{se}}` to the symmetric entropic
    affinity (SEA) problem.

    The algorithm computes the optimal dual variables
    :math:`\mathbf{\mu}^\star \in \mathbb{R}^n` and
    :math:`\mathbf{\varepsilon}^\star \in \mathbb{R}^n_{>0}` using dual ascent.
    The affinity matrix is then given by

    .. math::
        \forall (i,j), \: P^{\mathrm{se}}_{ij} = \exp \left( \frac{\mu^\star_{i} + \mu^\star_j - 2 C_{ij}}{\varepsilon^\star_i + \varepsilon^\star_j} \right) \:

    **Convex problem.** It amounts to the following convex optimization problem:

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

    It is a symmetric version of :class:`~torchdr.EntropicAffinity`,
    where a symmetry constraint is added in the optimization problem.

    .. note:: Unlike
        :math:`(\mathbf{P}^{\mathrm{e}} + (\mathbf{P}^{\mathrm{e}})^\top )/ 2` used in
        :class:`~torchdr.TSNE` where :math:`\mathbf{P}^{\mathrm{e}}` is the
        :class:`~torchdr.EntropicAffinity` matrix,
        :class:`~torchdr.affinity.SymmetricEntropicAffinity`
        allows to control the entropy and mass of each row/column of the
        affinity matrix.

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
    metric : str, optional
        Metric to use for computing distances, by default "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal of the distance matrix to 0.
    device : str, optional
        Device to use for computation.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        lr: float = 1e-1,
        eps_square: bool = True,
        tol: float = 1e-3,
        max_iter: int = 500,
        optimizer: str = "Adam",
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.perplexity = perplexity
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.eps_square = eps_square

    def _compute_log_affinity(self, X: torch.Tensor):
        r"""Solve the problem (SEA) in :cite:`van2024snekhorn`.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_samples, n_samples)
            Log of the symmetric entropic affinity matrix.
        """
        self.log_ = {}
        if self.verbose:
            print(
                "[TorchDR] Affinity : computing the Symmetric Entropic Affinity matrix."
            )

        C, _ = self._distance_matrix(X)

        n_samples_in = X.shape[0]
        perplexity = _check_perplexity(self.perplexity, n_samples_in, self.verbose)
        target_entropy = np.log(perplexity) + 1

        one = torch.ones(n_samples_in, dtype=X.dtype, device=X.device)

        # dual variables, size (n_samples)
        self.eps_ = torch.ones(n_samples_in, dtype=X.dtype, device=X.device)
        self.mu_ = torch.ones(n_samples_in, dtype=X.dtype, device=X.device)

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
                    "[TorchDR] ERROR Affinity: NaN in dual variables, "
                    "consider decreasing the learning rate."
                ),
            )

            log_affinity_matrix = _log_Pse(
                C, self.eps_, self.mu_, eps_square=self.eps_square
            )

        else:  # other optimizers including SGD and Adam
            optimizer = OPTIMIZERS[self.optimizer]([self.eps_, self.mu_], lr=self.lr)

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
                            f"[TorchDR] ERROR Affinity: NaN at iter {k}, "
                            "consider decreasing the learning rate."
                        ),
                    )

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
                                f"[TorchDR] Affinity : convergence reached at iter {k}."
                            )
                        break

                    if k == self.max_iter - 1 and self.verbose:
                        warnings.warn(
                            "[TorchDR] WARNING Affinity: max iter attained, "
                            "algorithm stops but may not have converged."
                        )

            self.n_iter_ = k
            log_affinity_matrix = log_P

        log_affinity_matrix -= math.log(n_samples_in)

        return log_affinity_matrix


class SinkhornAffinity(LogAffinity):
    r"""Compute the symmetric doubly stochastic affinity matrix.

    The algorithm computes the doubly stochastic matrix :math:`\mathbf{P}^{\mathrm{ds}}`
    with controlled global entropy using the symmetric Sinkhorn algorithm :cite:`sinkhorn1967concerning`.

    The algorithm computes the optimal dual variable
    :math:`\mathbf{f}^\star \in \mathbb{R}^n` such that

    .. math::
        \mathbf{P}^{\mathrm{ds}} \mathbf{1} = \mathbf{1} \quad \text{where} \quad \forall (i,j), \: P^{\mathrm{ds}}_{ij} = \exp(f^\star_i + f^\star_j - C_{ij} / \varepsilon) \:.

    where :

    - :math:`\mathbf{C}`: symmetric pairwise distance matrix between the samples.
    - :math:`\varepsilon`: entropic regularization parameter.
    - :math:`\mathbf{1} := (1,...,1)^\top`: all-ones vector.

    :math:`\mathbf{f}^\star` is computed by performing dual ascent via the Sinkhorn fixed-point iteration (eq. 25 in :cite:`feydy2019interpolating`).

    **Convex problem.** Consists in solving the following symmetric entropic optimal transport problem :cite:`cuturi2013sinkhorn`:

    .. math::
        \mathbf{P}^{\mathrm{ds}} \in \mathop{\arg\min}_{\mathbf{P} \in \mathcal{DS}} \: \langle \mathbf{C}, \mathbf{P} \rangle + \varepsilon \mathrm{H}(\mathbf{P})

    where :

    - :math:`\mathcal{DS} := \left\{ \mathbf{P} \in \mathbb{R}_+^{n \times n}: \: \mathbf{P} = \mathbf{P}^\top \:,\: \mathbf{P} \mathbf{1} = \mathbf{1} \right\}`: set of symmetric doubly stochastic matrices.
    - :math:`\mathrm{H}`: (global) Shannon entropy such that :math:`\mathrm{H}(\mathbf{P}) := - \sum_{ij} P_{ij} (\log P_{ij} - 1)`.

    **Bregman projection.** Another way to write this problem is to consider the
    KL projection of the Gaussian kernel
    :math:`\mathbf{K}_\varepsilon = \exp(- \mathbf{C} / \varepsilon)` onto the set
    of doubly stochastic matrices:

    .. math::
        \mathbf{P}^{\mathrm{ds}} = \mathrm{Proj}_{\mathcal{DS}}^{\mathrm{KL}}(\mathbf{K}_\varepsilon) := \mathop{\arg\min}_{\mathbf{P} \in \mathcal{DS}} \: \mathrm{KL}(\mathbf{P} \| \mathbf{K}_\varepsilon)

    where :math:`\mathrm{KL}(\mathbf{P} \| \mathbf{Q}) := \sum_{ij} P_{ij} (\log (Q_{ij} / P_{ij}) - 1) + Q_{ij}`
    is the Kullback Leibler divergence between :math:`\mathbf{P}` and
    :math:`\mathbf{Q}`.

    Parameters
    ----------
    eps : float, optional
        Regularization parameter for the Sinkhorn algorithm.
    tol : float, optional
        Precision threshold at which the algorithm stops.
    max_iter : int, optional
        Number of maximum iterations for the algorithm.
    base_kernel : {"gaussian", "student"}, optional
        Which base kernel to normalize as doubly stochastic.
    metric : str, optional
        Metric to use for computing distances (default "sqeuclidean").
    zero_diag : bool, optional
        Whether to set the diagonal of the distance matrix to 0.
    device : str, optional
        Device to use for computation.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity. Default is False.
    with_grad : bool, optional (default=False)
        If True, the Sinkhorn iterations are done with gradient tracking.
        If False, torch.no_grad() is used for the iterations.
    """  # noqa: E501

    def __init__(
        self,
        eps: float = 1.0,
        tol: float = 1e-5,
        max_iter: int = 1000,
        base_kernel: str = "gaussian",
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        with_grad: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.eps = eps
        self.tol = tol
        self.max_iter = max_iter
        self.base_kernel = base_kernel
        self.with_grad = with_grad

    def _compute_log_affinity(self, X: torch.Tensor, init_dual: torch.Tensor = None):
        r"""Compute the entropic doubly stochastic affinity matrix.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.
        init_dual : torch.Tensor of shape (n_samples), optional
            Initialization for the dual variable of the Sinkhorn algorithm.

        Returns
        -------
        log_affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_samples, n_samples)
            Log of the doubly stochastic affinity matrix.
        """
        C, _ = self._distance_matrix(X)
        if self.base_kernel == "student":
            C = (1 + C).log()

        if self.verbose:
            print(
                "[TorchDR] Affinity : computing the (KL) Doubly Stochastic "
                "Affinity matrix (Sinkhorn affinity)."
            )

        n_samples_in = C.shape[0]
        log_K = -C / self.eps

        # Performs warm-start if a dual variable f is provided
        self.dual_ = (
            torch.zeros(n_samples_in, dtype=X.dtype, device=X.device)
            if init_dual is None
            else init_dual
        )

        context_manager = (
            contextlib.nullcontext() if self.with_grad else torch.no_grad()
        )
        with context_manager:
            # Sinkhorn iterations
            for k in range(self.max_iter):
                # well conditioned symmetric Sinkhorn iteration (Feydy et al. 2019)
                reduction = -sum_matrix_vector(log_K, self.dual_).logsumexp(0).squeeze()
                self.dual_ = 0.5 * (self.dual_ + reduction)

                check_NaNs(
                    self.dual_, msg=f"[TorchDR] ERROR Affinity: NaN at iter {k}."
                )

                if torch.norm(self.dual_ - reduction) < self.tol:
                    if self.verbose:
                        print(f"[TorchDR] Affinity : convergence reached at iter {k}.")
                    break

                if k == self.max_iter - 1 and self.verbose:
                    warnings.warn(
                        "[TorchDR] WARNING Affinity: max iter attained, algorithm "
                        "stops but may not have converged."
                    )

        self.n_iter_ = k
        log_affinity_matrix = _log_Pds(log_K, self.dual_)

        log_affinity_matrix -= math.log(n_samples_in)

        return log_affinity_matrix


class NormalizedGaussianAffinity(LogAffinity):
    r"""Compute the Gaussian affinity matrix which can be normalized along a dimension.

    The algorithm computes :math:`\exp( - \mathbf{C} / \sigma)`
    where :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma` is the bandwidth parameter. The affinity can be normalized
    according to the specified normalization dimension.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth parameter.
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
        Verbosity.
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix. Default is (0, 1)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        normalization_dim: int | Tuple[int] = (0, 1),
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.sigma = sigma
        self.normalization_dim = normalization_dim

    def _compute_log_affinity(self, X: torch.Tensor):
        r"""Fit the normalized Gaussian affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        log_affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_samples, n_samples)
            Log of the normalized Gaussian affinity matrix.
        """
        C, _ = self._distance_matrix(X)

        log_affinity_matrix = -C / self.sigma

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                log_affinity_matrix, self.normalization_dim
            )
            log_affinity_matrix = log_affinity_matrix - self.log_normalization_

        if isinstance(self.normalization_dim, int):
            n_samples_in = X.shape[0]
            log_affinity_matrix -= math.log(n_samples_in)

        return log_affinity_matrix


class NormalizedStudentAffinity(LogAffinity):
    r"""Compute the Student affinity matrix which can be normalized along a dimension.

    Its expression is given by:

    .. math::
        \left(1 + \frac{\mathbf{C}}{\nu}\right)^{-\frac{\nu + 1}{2}}

    where :math:`\nu > 0` is the degrees of freedom parameter.
    The affinity can be normalized
    according to the specified normalization dimension.

    Parameters
    ----------
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
        Verbosity.
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix. Default is (0, 1)
    """

    def __init__(
        self,
        degrees_of_freedom: float = 1.0,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        normalization_dim: int | Tuple[int] = (0, 1),
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.degrees_of_freedom = degrees_of_freedom
        self.normalization_dim = normalization_dim

    def _compute_log_affinity(self, X: torch.Tensor):
        r"""Fits the normalized Student affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor of shape(n_samples, n_features)
            Input data.

        Returns
        -------
        log_affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            of shape(n_samples, n_samples)
            Log of the normalized Student affinity matrix.
        """
        C, _ = self._distance_matrix(X)

        log_affinity_matrix = (
            -0.5
            * (self.degrees_of_freedom + 1)
            * (C / self.degrees_of_freedom + 1).log()
        )

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                log_affinity_matrix, self.normalization_dim
            )
            log_affinity_matrix = log_affinity_matrix - self.log_normalization_

        if isinstance(self.normalization_dim, int):
            n_samples_in = X.shape[0]
            log_affinity_matrix -= math.log(n_samples_in)

        return log_affinity_matrix
