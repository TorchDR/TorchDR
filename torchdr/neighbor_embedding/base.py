# -*- coding: utf-8 -*-
"""
Base classes for Neighbor Embedding methods
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from torchdr.affinity import Affinity, LogAffinity
from torchdr.affinity_matcher import BatchedAffinityMatcher
from torchdr.utils import cross_entropy_loss


class NeighborEmbedding(BatchedAffinityMatcher):
    r"""
    Performs dimensionality reduction by solving the neighbor embedding problem.

    It amounts to solving the following optimization problem:

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity
        The affinity object for the output embedding space.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out fit_transform method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    optimizer : str, optional
        Optimizer to use for the optimization. Default is "Adam".
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e0.
    scheduler : str, optional
        Learning rate scheduler. Default is "constant".
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    init : str, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    tolog : bool, optional
        If True, logs the optimization process. Default is False.
    device : str, optional
        Device to use for computations. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for computations. Default is False.
    verbose : bool, optional
        Verbosity of the optimization process. Default is True.
    random_state : float, optional
        Random seed for reproducibility. Default is 0.
    coeff_attraction : float, optional
        Coefficient for the attraction term. Default is 1.0.
    coeff_repulsion : float, optional
        Coefficient for the repulsion term. Default is 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration. Default is None.
    batch_size : int or str, optional
        Batch size for the optimization. Default is None.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Affinity,
        kwargs_affinity_out: dict = {},
        n_components: int = 2,
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        lr: float = 1e0,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        tol: float = 1e-3,
        max_iter: int = 1000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        tolog: bool = False,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
        random_state: float = 0,
        coeff_attraction: float = 1.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = None,
        batch_size: int | str = None,
    ):

        if not isinstance(affinity_out, LogAffinity):
            raise ValueError(
                "[TorchDR] ERROR : in NeighborEmbedding, affinity_out must be "
                "a LogAffinity."
            )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            kwargs_affinity_out=kwargs_affinity_out,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            tol=tol,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
            batch_size=batch_size,
        )

        self.coeff_attraction = coeff_attraction
        self.coeff_repulsion = coeff_repulsion
        self.early_exaggeration_iter = early_exaggeration_iter

    def _additional_updates(self, step):
        if (  # stop early exaggeration phase
            self.coeff_attraction_ > 1 and step == self.early_exaggeration_iter
        ):
            self.coeff_attraction_ = 1
            # reinitialize optimizer and scheduler
            self._set_optimizer()
            self._set_scheduler()
        return self

    def _check_n_neighbors(self, n):
        param_list = ["perplexity", "n_neighbors"]

        for param_name in param_list:
            if hasattr(self, param_name):
                param_value = getattr(self, param_name)
                if n <= param_value:
                    if self.verbose:
                        print(
                            "[TorchDR] WARNING : Number of samples is smaller than "
                            f"{param_name} ({n} <= {param_value}), setting "
                            f"{param_name} to {n//2} (which corresponds to n//2)."
                        )
                    new_value = n // 2
                    setattr(self, param_name + "_", new_value)
                    setattr(self.affinity_in, param_name, new_value)

        return self

    def _fit(self, X: torch.Tensor):
        self._check_n_neighbors(X.shape[0])
        self.coeff_attraction_ = (
            self.coeff_attraction
        )  # coeff_attraction_ may change during the optimization

        super()._fit(X)

    def _repulsive_loss(self, log_Q):
        return 0

    def _loss(self):
        if self.batch_size is None:
            log_Q = self.affinity_out.fit_transform(self.embedding_, log=True)
            P = self.PX_

        else:
            P, log_Q = self.batched_affinity_in_out(log=True)

        attractive_term = cross_entropy_loss(P, log_Q, log=True)
        repulsive_term = self._repulsive_loss(log_Q)

        losses = (
            self.coeff_attraction_ * attractive_term
            + self.coeff_repulsion * repulsive_term
        )  # one loss per batch
        loss = losses.sum()
        return loss
