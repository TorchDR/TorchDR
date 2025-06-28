"""Affinity matcher base classes."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

import numpy as np
import torch

from torchdr.affinity import (
    Affinity,
    LogAffinity,
    SparseLogAffinity,
    UnnormalizedAffinity,
)
from torchdr.base import DRModule
from torchdr.utils import (
    check_NaNs,
    check_nonnegativity,
    cross_entropy_loss,
    square_loss,
    to_torch,
    ManifoldParameter,
    PoincareBallManifold,
    compile_if_requested,
)

from typing import Union, Dict, Optional, Any, Type


LOSS_DICT = {
    "square_loss": square_loss,
    "cross_entropy_loss": cross_entropy_loss,
}


class AffinityMatcher(DRModule):
    r"""Perform dimensionality reduction by matching two affinity matrices.

    It amounts to solving a problem of the form:

    .. math::

        \min_{\mathbf{Z}} \: \mathcal{L}( \mathbf{P}, \mathbf{Q})

    where :math:`\mathcal{L}` is a loss function, :math:`\mathbf{P}` is the
    input affinity matrix and :math:`\mathbf{Q}` is the affinity matrix of the
    embedding.

    The embedding optimization is performed using a first-order optimization method, with gradients calculated via PyTorch's automatic differentiation.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity, optional
        The affinity object for the output embedding space. Default is None.
        When None, a custom _loss method must be implemented.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    loss_fn : str, optional
        Loss function to use for the optimization. Default is "square_loss".
    kwargs_loss : dict, optional
        Additional keyword arguments for the loss function.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "Adam".
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    min_grad_norm : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    init : str, torch.Tensor, or np.ndarray, optional
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
    check_interval : int, optional
        Number of iterations between two checks for convergence. Default is 50.
    compile : bool, default=False
        Whether to use torch.compile for faster computation.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Optional[Affinity] = None,
        kwargs_affinity_out: Optional[Dict] = None,
        n_components: int = 2,
        loss_fn: str = "square_loss",
        kwargs_loss: Optional[Dict] = None,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        optimizer_kwargs: Optional[Dict] = None,
        lr: float = 1e0,
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        min_grad_norm: float = 1e-7,
        max_iter: int = 1000,
        init: Union[str, torch.Tensor, np.ndarray] = "pca",
        init_scaling: float = 1e-4,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        check_interval: int = 50,
        compile: bool = False,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            compile=compile,
            **kwargs,
        )

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.min_grad_norm = min_grad_norm
        self.check_interval = check_interval
        self.max_iter = max_iter
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        if loss_fn not in LOSS_DICT:
            raise ValueError(
                f"[TorchDR] ERROR : Loss function {loss_fn} not supported."
            )
        self.loss_fn = loss_fn
        self.kwargs_loss = kwargs_loss

        self.init = init
        self.init_scaling = init_scaling

        # --- check affinity_in ---
        if not isinstance(affinity_in, Affinity) and not affinity_in == "precomputed":
            raise ValueError(
                '[TorchDR] affinity_in must be an Affinity instance or "precomputed".'
            )
        self.affinity_in = affinity_in
        if isinstance(self.affinity_in, Affinity):
            self.affinity_in._pre_processed = True

        # --- check affinity_out ---
        if affinity_out is not None:
            if not isinstance(affinity_out, Affinity):
                raise ValueError(
                    "[TorchDR] ERROR : affinity_out must be an Affinity instance when not None."
                )
            if getattr(self.affinity_in, "sparsity", False) and not isinstance(
                affinity_out, UnnormalizedAffinity
            ):
                self.logger.warning(
                    "affinity_out must be a UnnormalizedAffinity "
                    "when affinity_in is sparse. Setting sparsity = False in affinity_in."
                )
                self.affinity_in.sparsity = False  # turn off sparsity
            affinity_out._pre_processed = True

        self.affinity_out = affinity_out
        self.kwargs_affinity_out = kwargs_affinity_out

        self.n_iter_ = -1

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        """Fit the model from data in X.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data.
        y : None
            Ignored.

        Returns
        -------
        embedding_ : torch.Tensor
            The embedding of the input data.
        """
        self.n_samples_in_, self.n_features_in_ = X.shape

        # --- Input affinity computation ---

        self.on_affinity_computation_start()

        # check if affinity_in is precomputed else compute it
        if self.affinity_in == "precomputed":
            if self.verbose:
                self.logger.info(
                    "[Stage 1/2] --- Using precomputed affinity matrix ---"
                )
            if self.n_features_in_ != self.n_samples_in_:
                raise ValueError(
                    '[TorchDR] ERROR : When affinity_in="precomputed" the input X '
                    "in fit must be a tensor of lazy tensor of shape "
                    "(n_samples, n_samples)."
                )
            check_nonnegativity(X)
            self.affinity_in_ = X
        else:
            if self.verbose:
                self.logger.info(
                    f"[Stage 1/2] --- Computing the input affinity matrix with {self.affinity_in.__class__.__name__} ---"
                )
            if isinstance(self.affinity_in, SparseLogAffinity):
                self.affinity_in_, self.NN_indices_ = self.affinity_in(
                    X, return_indices=True
                )
            else:
                self.affinity_in_ = self.affinity_in(X)

        self.on_affinity_computation_end()

        # --- Embedding optimization ---

        if self.verbose:
            self.logger.info("[Stage 2/2] --- Optimizing the embedding ---")

        self._init_embedding(X)
        self._set_params()
        self._set_learning_rate()
        self._set_optimizer()
        self._set_scheduler()

        grad_norm = float("nan")
        for step in range(self.max_iter):
            self.n_iter_ = step

            self.on_training_step_start()
            loss = self._training_step()
            self.on_training_step_end()

            check_NaNs(
                self.embedding_,
                msg="[TorchDR] ERROR AffinityMatcher : NaNs in the embeddings "
                f"at iter {step}.",
            )

            if self.verbose and (self.n_iter_ % self.check_interval == 0):
                lr = self.optimizer_.param_groups[0]["lr"]
                msg = (
                    f"Loss: {loss.item():.2e} | "
                    f"Grad norm: {grad_norm:.2e} | "
                    f"LR: {lr:.2e}"
                )
                self.logger.info(f"[{self.n_iter_}/{self.max_iter}] {msg}")

            check_convergence = self.n_iter_ % self.check_interval == 0
            if check_convergence:
                grad_norm = self.embedding_.grad.norm(2).item()
                if grad_norm < self.min_grad_norm:
                    if self.verbose:
                        self.logger.info(
                            f"Convergence reached at iter {self.n_iter_} with grad norm: "
                            f"{grad_norm:.2e}."
                        )
                    break

        return self.embedding_

    @compile_if_requested
    def _training_step(self):
        self.optimizer_.zero_grad(set_to_none=True)
        loss = self._loss()
        loss.backward()
        self.optimizer_.step()
        if self.scheduler_ is not None:
            self.scheduler_.step()
        return loss

    def _loss(self):
        if self.affinity_out is None:
            raise ValueError(
                "[TorchDR] ERROR : affinity_out is not set. Set it or implement _loss method."
            )

        if self.kwargs_affinity_out is None:
            self.kwargs_affinity_out = {}
        if self.kwargs_loss is None:
            self.kwargs_loss = {}

        # If cross entropy loss and affinity_out is LogAffinity, use log domain
        if (self.loss_fn == "cross_entropy_loss") and isinstance(
            self.affinity_out, LogAffinity
        ):
            self.kwargs_affinity_out.setdefault("log", True)
            self.kwargs_loss.setdefault("log", True)

        # If NN indices are available, restrict output affinity to NNs
        if getattr(self, "NN_indices_", None) is not None:
            self.kwargs_affinity_out.setdefault("indices", self.NN_indices_)

        Q = self.affinity_out(self.embedding_, **self.kwargs_affinity_out)

        loss = LOSS_DICT[self.loss_fn](self.affinity_in_, Q, **self.kwargs_loss)
        return loss

    def on_affinity_computation_start(self):
        pass

    def on_affinity_computation_end(self):
        pass

    def on_training_step_start(self):
        pass

    def on_training_step_end(self):
        pass

    def _set_params(self):
        self.params_ = [{"params": self.embedding_}]
        return self.params_

    def _set_optimizer(self):
        if isinstance(self.optimizer, str):
            # Try to get the optimizer from torch.optim
            try:
                optimizer_class = getattr(torch.optim, self.optimizer)
            except AttributeError:
                raise ValueError(
                    f"[TorchDR] ERROR: Optimizer '{self.optimizer}' not found in torch.optim."
                )
        else:
            if not issubclass(self.optimizer, torch.optim.Optimizer):
                raise ValueError(
                    "[TorchDR] ERROR: optimizer must be a string (name of an optimizer in "
                    "torch.optim) or a subclass of torch.optim.Optimizer."
                )
            optimizer_class = self.optimizer

        self.optimizer_ = optimizer_class(
            self.params_, lr=self.lr_, **(self.optimizer_kwargs or {})
        )
        return self.optimizer_

    def _set_learning_rate(self):
        if self.lr == "auto":
            if self.verbose:
                self.logger.warning(
                    "lr set to 'auto' without "
                    "any implemented rule. Setting lr=1.0 by default."
                )
            self.lr_ = 1.0
        else:
            self.lr_ = self.lr

    def _set_scheduler(self, n_iter: Optional[int] = None):
        n_iter = n_iter or self.max_iter

        if not hasattr(self, "optimizer_"):
            raise ValueError(
                "[TorchDR] ERROR : optimizer not set. "
                "Please call _set_optimizer before _set_scheduler."
            )

        # If scheduler is None, don't create a scheduler
        if self.scheduler is None:
            self.scheduler_ = None
            return self.scheduler_

        scheduler_kwargs = self.scheduler_kwargs or {}

        if isinstance(self.scheduler, str):
            # Try to get the scheduler from torch.optim.lr_scheduler
            try:
                scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler)
                self.scheduler_ = scheduler_class(self.optimizer_, **scheduler_kwargs)
            except AttributeError:
                raise ValueError(
                    f"[TorchDR] ERROR: Scheduler '{self.scheduler}' not found in torch.optim.lr_scheduler."
                )
        else:
            # Check if the scheduler is a subclass of LRScheduler
            if not issubclass(self.scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise ValueError(
                    "[TorchDR] ERROR: scheduler must be a string (name of a scheduler in "
                    "torch.optim.lr_scheduler) or a subclass of torch.optim.lr_scheduler.LRScheduler."
                )
            self.scheduler_ = self.scheduler(self.optimizer_, **scheduler_kwargs)

        return self.scheduler_

    def _init_embedding(self, X):
        n = X.shape[0]

        if isinstance(self.init, (torch.Tensor, np.ndarray)):
            embedding_ = to_torch(self.init, device=self.device)
            self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()

        elif self.init == "normal" or self.init == "random":
            embedding_ = torch.randn(
                (n, self.n_components),
                device=X.device if self.device == "auto" else self.device,
                dtype=X.dtype,
            )
            self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()

        elif self.init == "pca":
            from torchdr.spectral_embedding import PCA

            embedding_ = PCA(
                n_components=self.n_components, device=self.device
            ).fit_transform(X)
            self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()

        elif self.init == "hyperbolic":
            embedding_ = torch.randn(
                (n, self.n_components),
                device=X.device if self.device == "auto" else self.device,
                dtype=torch.float64,  # better double precision on hyperbolic manifolds
            )
            poincare_ball = PoincareBallManifold()
            embedding_ = self.init_scaling * embedding_
            self.embedding_ = ManifoldParameter(
                poincare_ball.expmap0(embedding_, c=1),
                requires_grad=True,
                manifold=poincare_ball,
                c=1,
            )

        else:
            raise ValueError(
                f"[TorchDR] ERROR : init {self.init} not supported in "
                f"{self.__class__.__name__}."
            )

        return self.embedding_.requires_grad_()
