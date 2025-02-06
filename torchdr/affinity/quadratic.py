"""Affinity matrices with quadratic constraints."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import warnings

import torch
from tqdm import tqdm

from torchdr.affinity import Affinity
from torchdr.utils import OPTIMIZERS, batch_transpose, check_NaNs, wrap_vectors


@wrap_vectors
def _Pds(C, dual, eps):
    r"""Return the quadratic doubly stochastic matrix from dual variable and cost.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        or shape (n_batch, batch_size, batch_size)
        Pairwise distance matrix.
    dual : torch.Tensor of shape (n) or (n_batch, batch_size)
        Dual variable of the normalization constraint.
    eps : float
        Dual variable of the quadratic constraint.

    Returns
    -------
    P : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        or shape (n_batch, batch_size, batch_size)
        Quadratic doubly stochastic affinity matrix.
    """
    dual_t = batch_transpose(dual)
    return (dual + dual_t - C).clamp(0, float("inf")) / eps


class DoublyStochasticQuadraticAffinity(Affinity):
    r"""Compute the symmetric doubly stochastic affinity.

    Implement the doubly stochastic normalized matrix with controlled
    global :math:`\ell_2` norm.

    The algorithm computes the optimal dual variable
    :math:`\mathbf{f}^\star \in \mathbb{R}^n` such that

    .. math::
        \mathbf{P}^{\star} \mathbf{1} = \mathbf{1} \quad \text{where} \quad \forall (i,j), \: P^{\star}_{ij} = \left[f^\star_i + f^\star_j - C_{ij} / \varepsilon \right]_{+} \:.

    :math:`\mathbf{f}^\star` is computed by performing dual ascent.

    **Convex problem.** Consists in solving the following symmetric quadratic
    optimal transport problem :cite:`zhang2023manifold`:

    .. math::
        \mathop{\arg\min}_{\mathbf{P} \in \mathcal{DS}} \: \langle \mathbf{C},
        \mathbf{P} \rangle + \varepsilon \| \mathbf{P} \|_2^2

    where :

    - :math:`\mathcal{DS} := \left\{ \mathbf{P} \in \mathbb{R}_+^{n \times n}: \: \mathbf{P} = \mathbf{P}^\top \:,\: \mathbf{P} \mathbf{1} = \mathbf{1} \right\}`: set of symmetric doubly stochastic matrices.
    - :math:`\mathbf{C}`: symmetric pairwise distance matrix between the samples.
    - :math:`\varepsilon`: quadratic regularization parameter.
    - :math:`\mathbf{1} := (1,...,1)^\top`: all-ones vector.

    **Bregman projection.** Another way to write this problem is to consider the
    :math:`\ell_2` projection of :math:`- \mathbf{C} / \varepsilon` onto the set of
    doubly stochastic matrices :math:`\mathcal{DS}`, as follows:

    .. math::
        \mathrm{Proj}_{\mathcal{DS}}^{\ell_2}(- \mathbf{C} / \varepsilon) := \mathop{\arg\min}_{\mathbf{P} \in \mathcal{DS}} \: \| \mathbf{P} + \mathbf{C} / \varepsilon \|_2 \:.

    Parameters
    ----------
    eps : float, optional
        Regularization parameter.
    init_dual : torch.Tensor of shape (n_samples), optional
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
    metric : str, optional
        Metric to use for computing distances (default "sqeuclidean").
    zero_diag : bool, optional
        Whether to set the diagonal elements of the affinity matrix to 0.
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
        eps: float = 1.0,
        init_dual: torch.Tensor = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        optimizer: str = "Adam",
        lr: float = 1e0,
        base_kernel: str = "gaussian",
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
        self.eps = eps
        self.init_dual = init_dual
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.lr = lr
        self.base_kernel = base_kernel

    def _compute_affinity(self, X: torch.Tensor):
        r"""Compute the quadratic doubly stochastic affinity matrix from input data X.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix.
        """
        if self.verbose:
            print(
                "[TorchDR] Affinity : computing the Doubly Stochastic Quadratic "
                "Affinity matrix."
            )

        C, _ = self._distance_matrix(X)
        if self.base_kernel == "student":
            C = (1 + C).log()

        n_samples_in = C.shape[0]
        one = torch.ones(n_samples_in, dtype=X.dtype, device=X.device)

        # Performs warm-start if an initial dual variable is provided
        self.dual_ = (
            torch.ones(n_samples_in, dtype=X.dtype, device=X.device)
            if self.init_dual is None
            else self.init_dual
        )

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
                warnings.warn(
                    "[TorchDR] Affinity (WARNING) : max iter attained, "
                    "algorithm stops but may not have converged."
                )

        self.n_iter_ = k
        affinity_matrix = _Pds(C, self.dual_, self.eps)

        affinity_matrix /= n_samples_in

        return affinity_matrix
