# -*- coding: utf-8 -*-
"""
Base classes for Neighbor Embedding methods
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
import warnings

from torchdr.affinity import (
    Affinity,
    LogAffinity,
    UnnormalizedAffinity,
    SparseLogAffinity,
)
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.utils import cross_entropy_loss, OPTIMIZERS


class NeighborEmbedding(AffinityMatcher):
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
        Additional keyword arguments for the affinity_out method.
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
        Tolerance for stopping criterion. Default is 1e-7.
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
        tol: float = 1e-7,
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
        **kwargs,
    ):

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
        )

        self.coeff_attraction = coeff_attraction
        self.coeff_repulsion = coeff_repulsion
        self.early_exaggeration_iter = early_exaggeration_iter

        # improve consistency with the sklearn API
        if "learning_rate" in kwargs:
            self.lr = kwargs["learning_rate"]
        if "min_grad_norm" in kwargs:
            self.tol = kwargs["min_grad_norm"]
        if "early_exaggeration" in kwargs:
            self.coeff_attraction = kwargs["early_exaggeration"]

    def _additional_updates(self, step):
        if (  # stop early exaggeration phase
            self.coeff_attraction_ > 1 and step == self.early_exaggeration_iter
        ):
            self.coeff_attraction_ = 1
            # reinitialize optimizer and scheduler
            self._set_learning_rate()
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
                        warnings.warn(
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

    def _loss(self):
        raise NotImplementedError("[TorchDR] ERROR : _loss method must be implemented.")

    def _set_learning_rate(self):
        if self.lr == "auto":
            if self.optimizer not in ["auto", "SGD"]:
                if self.verbose:
                    warnings.warn(
                        "[TorchDR] WARNING : when 'auto' is used for the learning "
                        "rate, the optimizer should be 'SGD'."
                    )
            # from the sklearn TSNE implementation
            self.lr_ = np.maximum(self.n_samples_in_ / self.coeff_attraction_ / 4, 50)
        else:
            self.lr_ = self.lr

    def _set_optimizer(self):
        optimizer = "SGD" if self.optimizer == "auto" else self.optimizer
        # from the sklearn TSNE implementation
        if self.optimizer_kwargs == "auto":
            if self.coeff_attraction_ > 1:
                optimizer_kwargs = {"momentum": 0.5}
            else:
                optimizer_kwargs = {"momentum": 0.8}
        else:
            optimizer_kwargs = self.optimizer_kwargs

        self.optimizer_ = OPTIMIZERS[optimizer](
            self.params_, lr=self.lr_, **(optimizer_kwargs or {})
        )
        return self.optimizer_


class SparseNeighborEmbedding(NeighborEmbedding):
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
        Additional keyword arguments for the affinity_out method.
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
        Tolerance for stopping criterion. Default is 1e-7.
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
        tol: float = 1e-7,
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
    ):
        # check affinity affinity_in
        if not isinstance(affinity_in, SparseLogAffinity):
            raise NotImplementedError(
                "[TorchDR] ERROR : when using SparseNeighborEmbedding, affinity_in "
                "must be a sparse affinity."
            )

        # check affinity affinity_out
        if not isinstance(affinity_out, UnnormalizedAffinity):
            raise NotImplementedError(
                "[TorchDR] ERROR : when using SparseNeighborEmbedding, affinity_out "
                "must be an UnnormalizedAffinity object."
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
            coeff_attraction=coeff_attraction,
            coeff_repulsion=coeff_repulsion,
            early_exaggeration_iter=early_exaggeration_iter,
        )

    def _attractive_loss(self):
        if isinstance(self.affinity_out, LogAffinity):
            log_Q = self.affinity_out(self.embedding_, log=True, indices=self.indices_)
            return cross_entropy_loss(self.PX_, log_Q, log=True)
        else:
            Q = self.affinity_out(self.embedding_, indices=self.indices_)
            return cross_entropy_loss(self.PX_, Q)

    def _repulsive_loss(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _repulsive_loss method must be implemented."
        )

    def _loss(self):
        loss = (
            self.coeff_attraction_ * self._attractive_loss()
            + self.coeff_repulsion * self._repulsive_loss()
        )
        return loss


class SampledNeighborEmbedding(SparseNeighborEmbedding):
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
        Additional keyword arguments for the affinity_out method.
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
        Tolerance for stopping criterion. Default is 1e-7.
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
    n_negatives : int, optional
        Number of negative samples for the repulsive loss.
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
        tol: float = 1e-7,
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
        n_negatives: int = 5,
    ):

        self.n_negatives = n_negatives

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
            coeff_attraction=coeff_attraction,
            coeff_repulsion=coeff_repulsion,
            early_exaggeration_iter=early_exaggeration_iter,
        )

    def _sample_negatives(self):
        if not hasattr(self, "n_negatives_"):
            if self.n_negatives > self.n_samples_in_:
                if self.verbose:
                    warnings.warn(
                        "[TorchDR] WARNING : n_negatives must be smaller than the "
                        f"number of samples. Here n_negatives={self.n_negatives} "
                        f"and n_samples_in={self.n_samples_in_}. Setting "
                        "n_negatives to n_samples_in."
                    )
                self.n_negatives_ = self.n_samples_in_
            else:
                self.n_negatives_ = self.n_negatives

        # For each point, uniformly sample n_negatives_ points
        # from the set of all other points.
        indices = self.generator_.integers(
            1, self.n_samples_in_, (self.n_samples_in_, self.n_negatives_)
        )
        indices = torch.from_numpy(indices)
        indices += (torch.arange(0, self.n_samples_in_))[:, None]
        indices = torch.remainder(indices, self.n_samples_in_)
        return indices
