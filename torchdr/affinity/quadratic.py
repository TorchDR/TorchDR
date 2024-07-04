# -*- coding: utf-8 -*-
"""
Affinity matrices with quadratic constraints
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from tqdm import tqdm
import numpy as np

from torchdr.affinity import Affinity
from torchdr.utils import OPTIMIZERS, wrap_vectors, check_NaNs, batch_transpose


@wrap_vectors
def _Pds(C, dual, eps):
    r"""
    Returns the quadratic doubly stochastic matrix P
    from the dual variable f and cost matrix C.
    """
    dual_t = batch_transpose(dual)
    return (dual + dual_t - C).clamp(0, float("inf")) / eps


class DoublyStochasticQuadraticAffinity(Affinity):
    r"""
    Computes the symmetric doubly stochastic affinity matrix with controlled
    global :math:`\ell_2` norm.
    Consists in solving the following symmetric quadratic optimal transport problem
    [10]_:

    .. math::
        \mathop{\arg\min}_{\mathbf{P} \in \mathcal{DS}} \: \langle \mathbf{C},
        \mathbf{P} \rangle + \varepsilon \| \mathbf{P} \|_2^2

    where :

    - :math:`\mathcal{DS} := \left\{ \mathbf{P} \in \mathbb{R}_+^{n \times n}: \: \mathbf{P} = \mathbf{P}^\top \:,\: \mathbf{P} \mathbf{1} = \mathbf{1} \right\}`: set of symmetric doubly stochastic matrices.
    - :math:`\mathbf{C}`: symmetric pairwise distance matrix between the samples.
    - :math:`\varepsilon`: quadratic regularization parameter.
    - :math:`\mathbf{1} := (1,...,1)^\top`: all-ones vector.

    The algorithm computes the optimal dual variable
    :math:`\mathbf{f}^\star \in \mathbb{R}^n` such that

    .. math::
        \mathbf{P}^{\star} \mathbf{1} = \mathbf{1} \quad \text{where} \quad \forall (i,j), \: P^{\star}_{ij} = \left[f^\star_i + f^\star_j - C_{ij} / \varepsilon \right]_{+} \:.

    :math:`\mathbf{f}^\star` is computed by performing dual ascent.

    **Bregman projection.** Another way to write this problem is to consider the
    :math:`\ell_2` projection of :math:`- \mathbf{C} / \varepsilon` onto the set of doubly stochastic matrices :math:`\mathcal{DS}`, as follows:

    .. math::
        \mathrm{Proj}_{\mathcal{DS}}^{\ell_2}(- \mathbf{C} / \varepsilon) := \mathop{\arg\min}_{\mathbf{P} \in \mathcal{DS}} \: \| \mathbf{P} + \mathbf{C} / \varepsilon \|_2 \:.

    Parameters
    ----------
    eps : float, optional
        Regularization parameter.
    init_dual : tensor of shape (n_samples), optional
        Initialization for the dual variable (default None).
    tol : float, optional
        Precision threshold at which the algorithm stops.
    max_iter : int, optional
        Number of maximum iterations for the algorithm.
    optimizer : {"Adam", "SGD", "NAdam"}, optional
        Optimizer to use for the dual ascent.
    lr : float, optional
        Learning rate for the optimizer.
    base_kernel : {"gaussian", "student"}, optional
        Which base kernel to normalize as doubly stochastic.
    tolog : bool, optional
        Whether to store intermediate result in a dictionary.
    metric : str, optional
        Metric to use for computing distances (default "euclidean").
    nodiag : bool, optional
        Whether to set the diagonal elements of the affinity matrix to 0.
    device : str, optional
        Device to use for computation.
    keops : bool, optional
        Whether to use KeOps for computation.
    verbose : bool, optional
        Verbosity.

    References
    ----------
    .. [10] Stephen Zhang, Gilles Mordant, Tetsuya Matsumoto, Geoffrey Schiebinger
        (2023). `Manifold Learning with Sparse Regularised Optimal Transport
        <https://arxiv.org/abs/2307.09816>`_. arXiv preprint.

    """  # noqa: E501

    def __init__(
        self,
        eps: float = 1.0,
        init_dual: torch.Tensor = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        optimizer: str = "Adam",
        lr: float = 1e0,
        base_kernel: str = "gaussian",
        tolog: bool = False,
        metric: str = "euclidean",
        nodiag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric, nodiag=nodiag, device=device, keops=keops, verbose=verbose
        )
        self.eps = eps
        self.init_dual = init_dual
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.lr = lr
        self.base_kernel = base_kernel
        self.tolog = tolog

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""Computes the quadratic doubly stochastic affinity matrix from input data X.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : DoublyStochasticQuadraticAffinity
            The fitted instance.
        """
        if self.verbose:
            print(
                "[TorchDR] Affinity : Computing the Doubly Stochastic Quadratic "
                "Affinity matrix."
            )
        super().fit(X)

        C = self._pairwise_distance_matrix(self.data_)
        if self.base_kernel == "student":
            C = (1 + C).log()

        self.n_samples_in_ = C.shape[0]
        one = torch.ones(
            self.n_samples_in_, dtype=self.data_.dtype, device=self.data_.device
        )

        # Performs warm-start if an initial dual variable is provided
        self.dual_ = (
            torch.ones(
                self.n_samples_in_, dtype=self.data_.dtype, device=self.data_.device
            )
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
                P = _Pds(C, self.dual_, self.eps)
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
        self.affinity_matrix_ = _Pds(C, self.dual_, self.eps)

        self.affinity_matrix_ /= self.n_samples_in_

        return self

    def get_batch(self, indices: torch.Tensor):
        r"""
        Extracts the affinity submatrix corresponding to the indices.

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
        if self.base_kernel == "student":
            C_batch = (1 + C_batch).log()

        dual_batch = self.dual_[indices]
        P_batch = _Pds(C_batch, dual_batch, self.eps)

        P_batch /= self.n_samples_in_
        return P_batch
