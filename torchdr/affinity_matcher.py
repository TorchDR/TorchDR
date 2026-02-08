"""Affinity matcher module."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torchdr.affinity import (
    Affinity,
    LogAffinity,
    SparseAffinity,
)
from torchdr.base import DRModule
from torchdr.distance import FaissConfig
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
    r"""Dimensionality reduction by matching two affinity matrices.

    Solves an optimization problem of the form:

    .. math::

        \min_{\mathbf{Z}} \: \mathcal{L}( \mathbf{P}, \mathbf{Q})

    where :math:`\mathcal{L}` is a loss function, :math:`\mathbf{P}` is the
    input affinity matrix and :math:`\mathbf{Q}` is the affinity matrix of the
    embedding.

    The embedding is optimized via first-order methods, with gradients computed
    either through PyTorch autograd or manually (when
    :attr:`_use_direct_gradients` is ``True``).

    When an :attr:`encoder` (neural network) is provided, its parameters are
    optimized instead of a raw embedding matrix, enabling out-of-sample
    extension via :meth:`transform`.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity, optional
        The affinity object for the output embedding space.
        When None, a custom :meth:`_compute_loss` method must be implemented.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    loss_fn : str, optional
        Loss function for optimization. Default is "square_loss".
    kwargs_loss : dict, optional
        Additional keyword arguments for the loss function.
    optimizer : str or torch.optim.Optimizer, optional
        Optimizer name from ``torch.optim`` or an optimizer class.
        Default is "Adam".
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer.
    lr : float or 'auto', optional
        Learning rate. Default is 1e0.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Scheduler name from ``torch.optim.lr_scheduler`` or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    min_grad_norm : float, optional
        Gradient norm threshold for convergence. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    init : str, torch.Tensor, or np.ndarray, optional
        Initialization for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    device : str, optional
        Device for computations. Default is "auto".
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Backend for handling sparsity and memory efficiency.
        Default is None (standard PyTorch).
    verbose : bool, optional
        Verbosity. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is None.
    check_interval : int, optional
        Iterations between convergence checks. Default is 50.
    compile : bool, default=False
        Whether to use ``torch.compile`` for faster computation.
    encoder : torch.nn.Module, optional
        Neural network mapping input data to the embedding space.
        Output dimension must match :attr:`n_components`.
        Default is None (optimize a raw embedding matrix).
    """

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
        backend: Union[str, FaissConfig, None] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        check_interval: int = 50,
        compile: bool = False,
        encoder: Optional[torch.nn.Module] = None,
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

        # --- Validate affinity_in ---
        if not isinstance(affinity_in, Affinity) and not affinity_in == "precomputed":
            raise ValueError(
                '[TorchDR] affinity_in must be an Affinity instance or "precomputed".'
            )
        self.affinity_in = affinity_in
        if isinstance(self.affinity_in, Affinity):
            self.affinity_in._pre_processed = True
            self.affinity_in.compile = self.compile

        # --- Validate affinity_out ---
        if affinity_out is not None:
            if not isinstance(affinity_out, Affinity):
                raise ValueError(
                    "[TorchDR] ERROR : affinity_out must be an Affinity instance "
                    "when not None."
                )
            affinity_out._pre_processed = True

        self.affinity_out = affinity_out
        self.kwargs_affinity_out = kwargs_affinity_out

        self.encoder = encoder

        self.n_iter_ = torch.tensor(-1, dtype=torch.long)

    # --- Fitting and optimization loop ---

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        """Compute the input affinity and optimize the embedding.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data (or ``(n_samples, n_samples)`` when
            ``affinity_in="precomputed"``).
        y : None
            Ignored.

        Returns
        -------
        embedding_ : torch.Tensor of shape (n_samples, n_components)
            The optimized embedding.
        """
        # --- Resolve metadata and device ---
        if isinstance(X, DataLoader):
            self.n_samples_in_ = len(X.dataset)
            for batch in X:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                self.n_features_in_ = batch.shape[1]
                self._dataloader_dtype_ = batch.dtype
                self._dataloader_device_ = batch.device
                break
            else:
                raise ValueError(
                    "[TorchDR] DataLoader is empty, cannot determine metadata. "
                    "Ensure DataLoader yields at least one batch."
                )
            self.device_ = (
                self._dataloader_device_ if self.device == "auto" else self.device
            )
        else:
            self.n_samples_in_, self.n_features_in_ = X.shape
            self.device_ = X.device if self.device == "auto" else self.device

        # --- Validate encoder ---
        if self.encoder is not None:
            if not isinstance(self.encoder, torch.nn.Module):
                raise TypeError("[TorchDR] encoder must be an nn.Module instance.")
            if isinstance(X, DataLoader):
                raise NotImplementedError(
                    "[TorchDR] encoder with DataLoader input is not yet supported."
                )
            with torch.no_grad():
                sample_out = self.encoder.to(device=self.device_, dtype=X.dtype)(X[:1])
                if sample_out.shape[-1] != self.n_components:
                    raise ValueError(
                        f"[TorchDR] encoder output dim ({sample_out.shape[-1]})"
                        f" != n_components ({self.n_components})."
                    )

        # --- Compute input affinity ---

        self.on_affinity_computation_start()

        if self.affinity_in == "precomputed":
            if self.verbose:
                self.logger.info("----- Using precomputed affinity matrix -----")
            if self.n_features_in_ != self.n_samples_in_:
                raise ValueError(
                    '[TorchDR] ERROR : When affinity_in="precomputed" the input '
                    "X in fit must be a tensor of lazy tensor of shape "
                    "(n_samples, n_samples)."
                )
            check_nonnegativity(X)
            self.register_buffer("affinity_in_", X, persistent=False)
        else:
            if self.verbose:
                self.logger.info(
                    "----- Computing the input affinity matrix with "
                    f"{self.affinity_in.__class__.__name__} -----"
                )
            if isinstance(self.affinity_in, SparseAffinity):
                affinity_matrix, nn_indices = self.affinity_in(X, return_indices=True)
                self.register_buffer("NN_indices_", nn_indices, persistent=False)
            else:
                affinity_matrix = self.affinity_in(X)

            # LazyTensors (keops) cannot be registered as buffers
            if self.affinity_in.backend == "keops":
                self.affinity_in_ = affinity_matrix
            else:
                self.register_buffer("affinity_in_", affinity_matrix, persistent=False)

        self.on_affinity_computation_end()

        # --- Optimize embedding ---

        if self.verbose:
            self.logger.info("----- Optimizing the embedding -----")

        self._init_embedding(X)
        self._set_params()
        self._set_learning_rate()
        self._configure_optimizer()
        self._configure_scheduler()

        # Free input data (not needed after init, unless encoder stores X_train_)
        if self.affinity_in != "precomputed" and self.encoder is None:
            del X
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        grad_norm = float("nan")
        for step in range(self.max_iter):
            self.n_iter_.fill_(step)

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
                msg_parts = []
                if loss is not None:
                    msg_parts.append(f"Loss: {loss.item():.2e}")
                msg_parts.append(f"Grad norm: {grad_norm:.2e}")
                msg_parts.append(f"LR: {lr:.2e}")
                msg = " | ".join(msg_parts)
                self.logger.info(f"[{self.n_iter_}/{self.max_iter}] {msg}")

            if self.n_iter_ % self.check_interval == 0:
                if self.encoder is not None:
                    grad_norm = (
                        sum(
                            p.grad.norm(2).item() ** 2
                            for p in self.encoder.parameters()
                            if p.grad is not None
                        )
                        ** 0.5
                    )
                else:
                    grad_norm = self.embedding_.grad.norm(2).item()
                if grad_norm < self.min_grad_norm:
                    if self.verbose:
                        self.logger.info(
                            f"Convergence reached at iter {self.n_iter_} "
                            f"with grad norm: {grad_norm:.2e}."
                        )
                    break

        self.clear_memory()
        return self.embedding_

    @compile_if_requested
    def _training_step(self):
        """Perform one optimization step (zero grad, loss/gradients, update).

        Two gradient modes are supported:

        **Direct gradients** (``_use_direct_gradients = True``): subclasses
        (e.g. UMAP) implement :meth:`_compute_gradients` which returns
        hand-derived embedding gradients. No loss scalar is computed.
        This is faster when closed-form gradients are available.

        **Autograd** (default): :meth:`_compute_loss` returns a scalar loss
        and ``loss.backward()`` computes gradients via PyTorch autograd.
        Used by TSNE, LargeVis, InfoTSNE, etc.

        In both modes the optimizer then steps using the accumulated gradients.
        """
        self.optimizer_.zero_grad(set_to_none=True)

        # When an encoder is used, the embedding is the encoder's output.
        # Gradients will flow back through the encoder via autograd (autograd
        # mode) or via embedding_.backward(gradient=...) (direct mode).
        if self.encoder is not None:
            self.embedding_ = self.encoder(self.X_train_)

        if getattr(self, "_use_direct_gradients", False):
            # --- Direct gradient mode ---
            # Subclass computes closed-form gradients w.r.t. the embedding.
            gradients = self._compute_gradients()
            if gradients is not None:
                if self.encoder is not None:
                    if getattr(self, "world_size", 1) > 1:
                        raise NotImplementedError(
                            "[TorchDR] encoder with distributed direct "
                            "gradients is not yet supported."
                        )
                    # Backprop through the encoder: embedding_ is the output
                    # of encoder(X), so .backward(gradient=...) applies the
                    # chain rule to update the encoder's parameters.
                    self.embedding_.backward(gradient=gradients)
                elif getattr(self, "world_size", 1) > 1:
                    # Distributed: each rank computes gradients for its chunk
                    # of the embedding. Assemble into full-size tensor and
                    # all-reduce so every rank has the complete gradient.
                    expected_chunk_size = len(self.chunk_indices_)
                    if gradients.shape[0] != expected_chunk_size:
                        raise RuntimeError(
                            f"Gradient size mismatch in distributed mode: "
                            f"expected {expected_chunk_size} gradients for "
                            f"chunk but _compute_gradients() returned "
                            f"{gradients.shape[0]}"
                        )
                    full_gradients = torch.zeros_like(self.embedding_)
                    chunk_start = self.chunk_indices_[0].item()
                    full_gradients[chunk_start : chunk_start + expected_chunk_size] = (
                        gradients
                    )
                    dist.all_reduce(full_gradients, op=dist.ReduceOp.SUM)
                    self.embedding_.grad = full_gradients
                else:
                    # Single-process, no encoder: assign gradients directly.
                    self.embedding_.grad = gradients
            loss = None
        else:
            # --- Autograd mode ---
            # Compute a scalar loss and let PyTorch differentiate it.
            loss = self._compute_loss()
            loss.backward()
            # Distributed: sum gradients across ranks.
            if getattr(self, "world_size", 1) > 1 and self.embedding_.grad is not None:
                dist.all_reduce(self.embedding_.grad, op=dist.ReduceOp.SUM)

        self.optimizer_.step()
        if self.scheduler_ is not None:
            self.scheduler_.step()
        return loss

    # --- Loss and gradient computation ---

    def _compute_loss(self):
        """Compute the loss between input and output affinities.

        Uses :attr:`loss_fn` to compare :attr:`affinity_in_` and the output
        affinity computed from :attr:`embedding_`. Subclasses can override
        this for custom loss structures.
        """
        if self.affinity_out is None:
            raise ValueError(
                "[TorchDR] ERROR : affinity_out is not set. "
                "Set it or implement _compute_loss method."
            )

        kwargs_affinity_out = dict(self.kwargs_affinity_out or {})
        kwargs_loss = dict(self.kwargs_loss or {})

        # Cross-entropy with LogAffinity: use log domain for numerical stability
        if (self.loss_fn == "cross_entropy_loss") and isinstance(
            self.affinity_out, LogAffinity
        ):
            kwargs_affinity_out.setdefault("log", True)
            kwargs_loss.setdefault("log", True)

        Q = self.affinity_out(self.embedding_, **kwargs_affinity_out)
        loss = LOSS_DICT[self.loss_fn](self.affinity_in_, Q, **kwargs_loss)
        return loss

    def _compute_gradients(self):
        """Compute embedding gradients manually.

        Must be implemented by subclasses that set
        ``_use_direct_gradients = True``.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : _compute_gradients method must be implemented "
            "when _use_direct_gradients is True."
        )

    # --- Lifecycle hooks ---
    # Override in subclasses to inject logic at specific stages of fitting.

    def on_affinity_computation_start(self):
        """Called before computing the input affinity matrix."""
        pass

    def on_affinity_computation_end(self):
        """Called after computing the input affinity matrix."""
        pass

    def on_training_step_start(self):
        """Called at the beginning of each optimization step."""
        pass

    def on_training_step_end(self):
        """Called at the end of each optimization step."""
        pass

    # --- Embedding initialization ---

    def _init_embedding(self, X):
        """Initialize the embedding from the :attr:`init` strategy or encoder.

        Supports ``"pca"``, ``"normal"``/``"random"``, ``"hyperbolic"``,
        or a user-provided tensor.
        """
        if not hasattr(self, "device_"):
            if isinstance(X, DataLoader):
                raise RuntimeError(
                    "[TorchDR] _init_embedding called with DataLoader before "
                    "_fit_transform. Call fit_transform() instead."
                )
            self.device_ = X.device if self.device == "auto" else self.device

        # Encoder: store training data and compute initial embedding
        if self.encoder is not None:
            self.register_buffer("X_train_", X, persistent=False)
            self.encoder.to(device=self.device_, dtype=X.dtype)
            with torch.no_grad():
                self.embedding_ = self.encoder(self.X_train_).detach()
            return self.embedding_

        if isinstance(X, DataLoader):
            n = self.n_samples_in_
            X_dtype = self._dataloader_dtype_
        else:
            n = X.shape[0]
            X_dtype = X.dtype

        if isinstance(self.init, (torch.Tensor, np.ndarray)):
            embedding_ = to_torch(self.init)
            embedding_ = embedding_.to(device=self.device_, dtype=X_dtype)
            self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()

        elif self.init in ("normal", "random"):
            embedding_ = torch.randn(
                (n, self.n_components),
                device=self.device_,
                dtype=X_dtype,
            )
            self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()

        elif self.init == "pca":
            if isinstance(X, DataLoader):
                from torchdr.spectral_embedding import IncrementalPCA

                embedding_ = IncrementalPCA(
                    n_components=self.n_components, device=self.device
                ).fit_transform(X)
            else:
                from torchdr.spectral_embedding.pca import PCA

                embedding_ = PCA(
                    n_components=self.n_components, device=self.device
                ).fit_transform(X)
            if embedding_.device != self.device_:
                embedding_ = embedding_.to(self.device_)
            self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()

        elif self.init == "hyperbolic":
            embedding_ = torch.randn(
                (n, self.n_components),
                device=self.device_,
                dtype=torch.float64,
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

    # --- Optimizer and scheduler configuration ---

    def _set_params(self):
        """Set the parameters to optimize (encoder weights or embedding)."""
        if self.encoder is not None:
            self.params_ = [{"params": self.encoder.parameters()}]
        else:
            self.params_ = [{"params": self.embedding_}]
        return self.params_

    def _set_learning_rate(self):
        """Resolve the learning rate (handles ``lr='auto'``)."""
        if self.lr == "auto":
            if self.verbose:
                self.logger.warning(
                    "lr set to 'auto' without "
                    "any implemented rule. Setting lr=1.0 by default."
                )
            self.lr_ = 1.0
        else:
            self.lr_ = self.lr

    def _configure_optimizer(self):
        """Instantiate the optimizer from :attr:`optimizer`."""
        if isinstance(self.optimizer, str):
            try:
                optimizer_class = getattr(torch.optim, self.optimizer)
            except AttributeError:
                raise ValueError(
                    f"[TorchDR] ERROR: Optimizer '{self.optimizer}' not found "
                    "in torch.optim."
                )
        else:
            if not issubclass(self.optimizer, torch.optim.Optimizer):
                raise ValueError(
                    "[TorchDR] ERROR: optimizer must be a string (name of an "
                    "optimizer in torch.optim) or a subclass of "
                    "torch.optim.Optimizer."
                )
            optimizer_class = self.optimizer

        self.optimizer_ = optimizer_class(
            self.params_,
            lr=torch.tensor(self.lr_),
            **(self.optimizer_kwargs or {}),
        )
        return self.optimizer_

    def _configure_scheduler(self, n_iter: Optional[int] = None):
        """Instantiate the learning rate scheduler from :attr:`scheduler`."""
        n_iter = n_iter or self.max_iter

        if not hasattr(self, "optimizer_"):
            raise ValueError(
                "[TorchDR] ERROR : optimizer not set. "
                "Please call _configure_optimizer before _configure_scheduler."
            )

        if self.scheduler is None:
            self.scheduler_ = None
            return self.scheduler_

        scheduler_kwargs = self.scheduler_kwargs or {}

        if isinstance(self.scheduler, str):
            try:
                scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler)
                self.scheduler_ = scheduler_class(self.optimizer_, **scheduler_kwargs)
            except AttributeError:
                raise ValueError(
                    f"[TorchDR] ERROR: Scheduler '{self.scheduler}' not found "
                    "in torch.optim.lr_scheduler."
                )
        else:
            if not issubclass(self.scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise ValueError(
                    "[TorchDR] ERROR: scheduler must be a string (name of a "
                    "scheduler in torch.optim.lr_scheduler) or a subclass of "
                    "torch.optim.lr_scheduler.LRScheduler."
                )
            self.scheduler_ = self.scheduler(self.optimizer_, **scheduler_kwargs)

        return self.scheduler_

    # --- Memory management ---

    def clear_memory(self):
        """Clear training-related state (affinities, optimizer, scheduler)."""
        super().clear_memory()

        # LazyTensors (keops) are not buffers and must be deleted explicitly
        if hasattr(self, "affinity_in_") and isinstance(self.affinity_in, Affinity):
            if self.affinity_in.backend == "keops":
                delattr(self, "affinity_in_")

        if isinstance(self.affinity_in, Affinity):
            self.affinity_in.clear_memory()
        if isinstance(self.affinity_out, Affinity):
            self.affinity_out.clear_memory()

        for attr in ["optimizer_", "scheduler_", "params_", "lr_"]:
            if hasattr(self, attr):
                delattr(self, attr)
