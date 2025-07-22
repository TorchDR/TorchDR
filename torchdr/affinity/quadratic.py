"""Affinity matrices with quadratic constraints."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Optional

import torch

from torchdr.affinity import Affinity
from torchdr.utils import (
    matrix_transpose,
    check_NaNs,
    wrap_vectors,
    compile_if_requested,
)


@wrap_vectors
def _Pds(C, dual, eps):
    dual_t = matrix_transpose(dual)
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
    check_interval : int, optional
        Interval for logging progress.
    optimizer : str, optional
        Optimizer to use for the dual ascent (default 'Adam').
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
    compile : bool, optional
        Whether to compile the computation. Default is False.
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        eps: float = 1.0,
        init_dual: Optional[torch.Tensor] = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        check_interval: int = 50,
        optimizer: str = "Adam",
        lr: float = 1e0,
        base_kernel: str = "gaussian",
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            compile=compile,
            _pre_processed=_pre_processed,
        )
        self.eps = eps
        self.init_dual = init_dual
        self.tol = tol
        self.max_iter = max_iter
        self.check_interval = check_interval
        self.optimizer = optimizer
        self.lr = lr
        self.base_kernel = base_kernel
        self.n_iter_ = torch.tensor(0, dtype=torch.long)

    @compile_if_requested
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
        C = self._distance_matrix(X)
        if self.base_kernel == "student":
            C = (1 + C).log()

        n_samples_in = C.shape[0]
        one = torch.ones(n_samples_in, dtype=X.dtype, device=X.device)

        # Performs warm-start if an initial dual variable is provided
        dual = (
            torch.ones(n_samples_in, dtype=X.dtype, device=X.device)
            if self.init_dual is None
            else self.init_dual
        )
        self.register_buffer("dual_", dual, persistent=False)

        optimizer_class = getattr(torch.optim, self.optimizer)
        optimizer = optimizer_class([self.dual_], lr=self.lr)

        # Dual ascent iterations
        for k in range(self.max_iter):
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

            if self.verbose and (k % self.check_interval == 0):
                P_sum_mean = float(P_sum.mean().item())
                P_sum_std = float(P_sum.std().item())
                msg = f"Marginal:{P_sum_mean: .2e} (std:{P_sum_std: .2e})"
                self.logger.info(f"[{k}/{self.max_iter}] {msg}")

            if torch.norm(grad_dual) < self.tol:
                if self.verbose:
                    self.logger.info(f"Convergence reached at iter {k}.")
                break

            if k == self.max_iter - 1 and self.verbose:
                self.logger.warning(
                    "Max iter attained, algorithm stops but may not have converged."
                )

        self.n_iter_ = k
        affinity_matrix = _Pds(C, self.dual_, self.eps)

        affinity_matrix /= n_samples_in  # sum of each row is 1/n so that total sum is 1

        return affinity_matrix
