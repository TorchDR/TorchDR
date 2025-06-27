"""Base classes for Neighbor Embedding methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import warnings
import numpy as np
from typing import Any, Dict, Union, Optional, Type
import torch

from torchdr.affinity import (
    Affinity,
    SparseLogAffinity,
    UnnormalizedAffinity,
    UnnormalizedLogAffinity,
)
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.utils import cross_entropy_loss


class NeighborEmbedding(AffinityMatcher):
    r"""Solves the neighbor embedding problem.

    It amounts to solving:

    .. math::

        \min_{\mathbf{Z}} \: - \lambda \sum_{ij} P_{ij} \log Q_{ij} + \mathcal{L}_{\mathrm{rep}}(\mathbf{Q})

    where :math:`\mathbf{P}` is the input affinity matrix, :math:`\mathbf{Q}` is the
    output affinity matrix, :math:`\mathcal{L}_{\mathrm{rep}}` is the repulsive
    term of the loss function, :math:`\lambda` is the :attr:`early_exaggeration_coeff`
    parameter.

    Note that the early exaggeration coefficient :math:`\lambda` is set to
    :math:`1` after the early exaggeration phase which duration is controlled by the
    :attr:`early_exaggeration_iter` parameter.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity, optional
        The affinity object for the output embedding space. Default is None.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD". For best results, we recommend using "SGD" with 'auto' learning rate.
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto',
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    min_grad_norm : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 2000.
    init : str or torch.Tensor or np.ndarray, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    device : str, optional
        Device to use for computations. Default is "auto".
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is None.
    early_exaggeration_coeff : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        Default is None.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration. Default is None.
    check_interval : int, optional
        Number of iterations between two checks for convergence. Default is 50.
    jit_compile : bool, default=False
        Whether to compile the loss function with `torch.compile` for faster
        computation.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Optional[Affinity] = None,
        kwargs_affinity_out: Optional[Dict] = None,
        n_components: int = 2,
        lr: Union[float, str] = 1e0,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = "auto",
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        init: Union[str, torch.Tensor, np.ndarray] = "pca",
        init_scaling: float = 1e-4,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: Optional[float] = None,
        early_exaggeration_iter: Optional[int] = None,
        check_interval: int = 50,
        jit_compile: bool = False,
        **kwargs: Any,
    ):
        self.early_exaggeration_iter = early_exaggeration_iter
        if self.early_exaggeration_iter is None:
            self.early_exaggeration_iter = 0
        self.early_exaggeration_coeff = early_exaggeration_coeff
        if self.early_exaggeration_coeff is None:
            self.early_exaggeration_coeff = 1

        # improve consistency with the sklearn API
        if "learning_rate" in kwargs:
            self.lr = kwargs["learning_rate"]
        if "min_grad_norm" in kwargs:
            self.min_grad_norm = kwargs["min_grad_norm"]
        if "early_exaggeration" in kwargs:
            self.early_exaggeration_coeff = kwargs["early_exaggeration"]

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
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            check_interval=check_interval,
            jit_compile=jit_compile,
        )

    def on_training_step_end(self):
        if (  # stop early exaggeration phase
            self.early_exaggeration_coeff_ > 1
            and self.n_iter_ == self.early_exaggeration_iter
        ):
            self.early_exaggeration_coeff_ = 1
            # reinitialize optim
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
                    raise ValueError(
                        f"[TorchDR] ERROR : Number of samples is smaller than {param_name} "
                        f"({n} <= {param_value})."
                    )

        return self

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        self._check_n_neighbors(X.shape[0])
        self.early_exaggeration_coeff_ = (
            self.early_exaggeration_coeff
        )  # early_exaggeration_ may change during the optimization

        return super()._fit_transform(X, y)

    def _loss(self):
        raise NotImplementedError("[TorchDR] ERROR : _loss method must be implemented.")

    def _set_learning_rate(self):
        if self.lr == "auto":
            if self.optimizer != "SGD":
                if self.verbose:
                    warnings.warn(
                        "[TorchDR] WARNING : when 'auto' is used for the learning "
                        "rate, the optimizer should be 'SGD'."
                    )
            # from the sklearn TSNE implementation
            self.lr_ = max(self.n_samples_in_ / self.early_exaggeration_coeff_ / 4, 50)
        else:
            self.lr_ = self.lr

    def _set_optimizer(self):
        if isinstance(self.optimizer, str):
            # Get optimizer directly from torch.optim
            try:
                optimizer_class = getattr(torch.optim, self.optimizer)
            except AttributeError:
                raise ValueError(
                    f"[TorchDR] ERROR: Optimizer '{self.optimizer}' not found in torch.optim"
                )
        else:
            if not issubclass(self.optimizer, torch.optim.Optimizer):
                raise ValueError(
                    "[TorchDR] ERROR: optimizer must be a string (name of an optimizer in "
                    "torch.optim) or a subclass of torch.optim.Optimizer"
                )
            # Assume it's already an optimizer class
            optimizer_class = self.optimizer

        # If 'auto' and SGD, set momentum based on early exaggeration phase
        if self.optimizer_kwargs == "auto":
            if self.optimizer == "SGD":
                if self.early_exaggeration_coeff_ > 1:
                    optimizer_kwargs = {"momentum": 0.5}
                else:
                    optimizer_kwargs = {"momentum": 0.8}
            else:
                optimizer_kwargs = {}
        else:
            optimizer_kwargs = self.optimizer_kwargs or {}

        self.optimizer_ = optimizer_class(self.params_, lr=self.lr_, **optimizer_kwargs)
        return self.optimizer_

    def _set_scheduler(self):
        if self.early_exaggeration_coeff_ > 1:
            n_iter = min(self.early_exaggeration_iter, self.max_iter)
        else:
            n_iter = self.max_iter - self.early_exaggeration_iter
        super()._set_scheduler(n_iter)


class SparseNeighborEmbedding(NeighborEmbedding):
    r"""Solves the neighbor embedding problem with a sparse input affinity matrix.

    It amounts to solving:

    .. math::

        \min_{\mathbf{Z}} \: - \lambda \sum_{ij} P_{ij} \log Q_{ij} + \mathcal{L}_{\mathrm{rep}}( \mathbf{Q})

    where :math:`\mathbf{P}` is the input affinity matrix, :math:`\mathbf{Q}` is the
    output affinity matrix, :math:`\mathcal{L}_{\mathrm{rep}}` is the repulsive
    term of the loss function, :math:`\lambda` is the :attr:`early_exaggeration_coeff`
    parameter.

    **Fast attraction.** This class should be used when the input affinity matrix is a :class:`~torchdr.SparseLogAffinity` and the output affinity matrix is an :class:`~torchdr.UnnormalizedAffinity`. In such cases, the attractive term can be computed with linear complexity.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity, optional
        The affinity object for the output embedding space. Default is None.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD". For best results, we recommend using "SGD" with 'auto' learning rate.
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto',
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    min_grad_norm : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 2000.
    init : str or torch.Tensor or np.ndarray, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    device : str, optional
        Device to use for computations. Default is "auto".
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is None.
    early_exaggeration_coeff : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        Default is 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration. Default is None.
    check_interval : int, optional
        Number of iterations between two checks for convergence. Default is 50.
    jit_compile : bool, default=False
        Whether to compile the loss function with `torch.compile` for faster
        computation.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Optional[Affinity] = None,
        kwargs_affinity_out: Optional[Dict] = None,
        n_components: int = 2,
        lr: Union[float, str] = 1e0,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = "auto",
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        init: Union[str, torch.Tensor, np.ndarray] = "pca",
        init_scaling: float = 1e-4,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 1.0,
        early_exaggeration_iter: Optional[int] = None,
        check_interval: int = 50,
        jit_compile: bool = False,
    ):
        # check affinity affinity_in
        if not isinstance(affinity_in, SparseLogAffinity):
            raise NotImplementedError(
                "[TorchDR] ERROR : when using SparseNeighborEmbedding, affinity_in "
                "must be a sparse affinity."
            )

        # check affinity affinity_out (only if not None)
        if affinity_out is not None and not isinstance(
            affinity_out, UnnormalizedAffinity
        ):
            raise NotImplementedError(
                "[TorchDR] ERROR : when using SparseNeighborEmbedding, affinity_out "
                "must be an UnnormalizedAffinity object or None."
            )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            kwargs_affinity_out=kwargs_affinity_out,
            n_components=n_components,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            early_exaggeration_coeff=early_exaggeration_coeff,
            early_exaggeration_iter=early_exaggeration_iter,
            check_interval=check_interval,
            jit_compile=jit_compile,
        )

    def _attractive_loss(self):
        if isinstance(self.affinity_out, UnnormalizedLogAffinity):
            log_Q = self.affinity_out(
                self.embedding_, log=True, indices=self.NN_indices_
            )
            return cross_entropy_loss(self.affinity_in_, log_Q, log=True)
        else:
            Q = self.affinity_out(self.embedding_, indices=self.NN_indices_)
            return cross_entropy_loss(self.affinity_in_, Q)

    def _repulsive_loss(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _repulsive_loss method must be implemented."
        )

    def _loss(self):
        loss = (
            self.early_exaggeration_coeff_ * self._attractive_loss()
            + self._repulsive_loss()
        )
        return loss


class SampledNeighborEmbedding(SparseNeighborEmbedding):
    r"""Solves the neighbor embedding problem with both sparsity and sampling.

    It amounts to solving:

    .. math::

        \min_{\mathbf{Z}} \: - \lambda \sum_{ij} P_{ij} \log Q_{ij} + \mathcal{L}_{\mathrm{rep}}( \mathbf{Q})

    where :math:`\mathbf{P}` is the input affinity matrix, :math:`\mathbf{Q}` is the
    output affinity matrix, :math:`\mathcal{L}_{\mathrm{rep}}` is the repulsive
    term of the loss function, :math:`\lambda` is the :attr:`early_exaggeration_coeff`
    parameter.

    **Fast attraction.** This class should be used when the input affinity matrix is a
    :class:`~torchdr.SparseLogAffinity` and the output affinity matrix is an
    :class:`~torchdr.UnnormalizedAffinity`. In such cases, the attractive term
    can be computed with linear complexity.

    **Fast repulsion.** A stochastic estimation of the repulsive term is used
    to reduce its complexity to linear.
    This is done by sampling a fixed number of negative samples
    :attr:`n_negatives` for each point.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity, optional
        The affinity object for the output embedding space. Default is None.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD". For best results, we recommend using "SGD" with 'auto' learning rate.
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto',
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    min_grad_norm : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 2000.
    init : str, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    device : str, optional
        Device to use for computations. Default is "auto".
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is None.
    early_exaggeration_coeff : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        Default is 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration. Default is None.
    n_negatives : int, optional
        Number of negative samples for the repulsive loss.
    check_interval : int, optional
        Number of iterations between two checks for convergence. Default is 50.
    jit_compile : bool, default=False
        Whether to compile the loss function with `torch.compile` for faster
        computation.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Optional[Affinity] = None,
        kwargs_affinity_out: Optional[Dict] = None,
        n_components: int = 2,
        lr: Union[float, str] = 1e0,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = "auto",
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 1.0,
        early_exaggeration_iter: Optional[int] = None,
        n_negatives: int = 5,
        check_interval: int = 50,
        discard_NNs: bool = True,
        jit_compile: bool = False,
    ):
        self.n_negatives = n_negatives
        self.discard_NNs = discard_NNs

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            kwargs_affinity_out=kwargs_affinity_out,
            n_components=n_components,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            early_exaggeration_coeff=early_exaggeration_coeff,
            early_exaggeration_iter=early_exaggeration_iter,
            check_interval=check_interval,
            jit_compile=jit_compile,
        )

    def on_affinity_computation_end(self):
        device = getattr(self.NN_indices_, "device", "cpu")
        self_idxs = torch.arange(self.n_samples_in_, device=device).unsqueeze(1)

        if self.discard_NNs:
            if not hasattr(self, "NN_indices_"):
                self.logger.warning(
                    "NN_indices_ not found. Cannot discard NNs from negative sampling."
                )
                exclude = self_idxs
            else:
                exclude = torch.cat([self_idxs, self.NN_indices_], dim=1)
        else:
            exclude = self_idxs
        self.exclude_, _ = exclude.sort(dim=1)

        n_possible = self.n_samples_in_ - self.exclude_.shape[1]
        if self.n_negatives > n_possible and self.verbose:
            raise ValueError(
                f"[TorchDR] ERROR : requested {self.n_negatives} negatives but "
                f"only {n_possible} available."
            )

    def on_training_step_start(self):
        # Sample negatives
        negatives = torch.randint(
            1,
            self.n_samples_in_ - self.exclude_.shape[1],
            (self.n_samples_in_, self.n_negatives),
            device=self.embedding_.device,
        )
        shifts = torch.searchsorted(self.exclude_, negatives, right=True)
        self.neg_indices_ = negatives + shifts
