"""Base classes for Neighbor Embedding methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import warnings
import os
import numpy as np
from typing import Any, Dict, Union, Optional, Type
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torchdr.affinity import Affinity
from torchdr.distance import FaissConfig
from torchdr.affinity_matcher import AffinityMatcher


class NeighborEmbedding(AffinityMatcher):
    r"""Base class for neighbor embedding methods.

    All neighbor embedding methods solve an optimization problem of the form:

    .. math::

        \min_{\mathbf{Z}} \: - \lambda \sum_{ij} P_{ij} \log Q_{ij} + \rho \cdot \mathcal{L}_{\mathrm{rep}}(\mathbf{Q})

    where :math:`\mathbf{P}` is the input affinity matrix, :math:`\mathbf{Q}` is the
    output affinity matrix, :math:`\lambda` is the early exaggeration coefficient,
    :math:`\rho` is :attr:`repulsion_strength`, and
    :math:`\mathcal{L}_{\mathrm{rep}}` is a repulsive term that prevents collapse.

    This class extends :class:`~torchdr.AffinityMatcher` with functionality
    specific to neighbor embedding:

    - **Loss decomposition**: By default, the loss is decomposed into an
      attractive term and a repulsive term via :meth:`_compute_attractive_loss`
      and :meth:`_compute_repulsive_loss`. When :attr:`_use_direct_gradients` is
      ``True``, subclasses implement :meth:`_compute_attractive_gradients` and
      :meth:`_compute_repulsive_gradients` instead. Subclasses that need a
      different loss structure can override :meth:`_compute_loss` directly.
    - **Early exaggeration**: The attraction term is scaled by
      :attr:`early_exaggeration_coeff` (:math:`\lambda`) for the first
      :attr:`early_exaggeration_iter` iterations to encourage cluster formation.
    - **Auto learning rate**: When ``lr='auto'``, the learning rate is set
      adaptively based on the number of samples.
    - **Auto optimizer tuning**: When ``optimizer_kwargs='auto'`` with SGD,
      momentum is adjusted between the early exaggeration and normal phases.
    - **Distributed multi-GPU training**: When launched with ``torchrun``,
      this class partitions the input affinity across GPUs, broadcasts the
      embedding, and synchronizes gradients via all-reduce. Set
      ``distributed='auto'`` (default) to auto-detect.

    .. note::
        The default values for ``lr='auto'``, ``optimizer_kwargs='auto'``, and
        early exaggeration are based on the t-SNE paper
        :cite:`van2008visualizing` and its scikit-learn implementation. These
        defaults work well for t-SNE but may need tuning for other methods.

    **Direct subclasses**: :class:`TSNE`, :class:`SNE`, :class:`COSNE`
    (compute the repulsive term exactly), :class:`TSNEkhorn` (overrides the
    full loss), :class:`NegativeSamplingNeighborEmbedding` (approximates
    the repulsive term via sampling).

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
        Default is None.
    scheduler_kwargs : dict, 'auto', or None, optional
        Additional keyword arguments for the scheduler. Default is 'auto', which
        corresponds to a linear decay from the learning rate to 0 for `LinearLR`.
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
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
        Default is None.
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is None.
    early_exaggeration_coeff : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        Default is None (no early exaggeration).
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration. Default is None.
    repulsion_strength: float, optional
        Strength of the repulsive term. Default is 1.0.
    check_interval : int, optional
        Number of iterations between two checks for convergence. Default is 50.
    compile : bool, default=False
        Whether to use torch.compile for faster computation.
    distributed : bool or 'auto', optional
        Whether to use distributed computation across multiple GPUs.
        - "auto": Automatically detect if running with torchrun (default)
        - True: Force distributed mode (requires torchrun)
        - False: Disable distributed mode
        Default is "auto".
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
        scheduler_kwargs: Union[Dict, str, None] = "auto",
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        init: Union[str, torch.Tensor, np.ndarray] = "pca",
        init_scaling: float = 1e-4,
        device: str = "auto",
        backend: Union[str, FaissConfig, None] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: Optional[float] = None,
        early_exaggeration_iter: Optional[int] = None,
        repulsion_strength: float = 1.0,
        check_interval: int = 50,
        compile: bool = False,
        distributed: Union[bool, str] = "auto",
        **kwargs: Any,
    ):
        self.early_exaggeration_iter = early_exaggeration_iter
        if self.early_exaggeration_iter is None:
            self.early_exaggeration_iter = 0
        self.early_exaggeration_coeff = early_exaggeration_coeff
        if self.early_exaggeration_coeff is None:
            self.early_exaggeration_coeff = 1

        self.repulsion_strength = repulsion_strength

        # improve consistency with the sklearn API
        if "learning_rate" in kwargs:
            self.lr = kwargs.pop("learning_rate")
        if "early_exaggeration" in kwargs:
            self.early_exaggeration_coeff = kwargs.pop("early_exaggeration")

        # by default, the linear scheduler goes from 1 to 0
        _scheduler_kwargs = scheduler_kwargs
        if scheduler == "LinearLR" and scheduler_kwargs == "auto":
            _scheduler_kwargs = {
                "start_factor": torch.tensor(1.0),
                "end_factor": torch.tensor(0),
                "total_iters": max_iter,
            }

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            kwargs_affinity_out=kwargs_affinity_out,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=_scheduler_kwargs,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            check_interval=check_interval,
            compile=compile,
            **kwargs,
        )

        self._setup_distributed(distributed)

    # --- Loss decomposition (attractive + repulsive) ---
    # Subclasses must implement _compute_attractive_loss and _compute_repulsive_loss.
    # Alternatively, subclasses can override _compute_loss directly (e.g. TSNEkhorn).

    def _compute_attractive_loss(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _compute_attractive_loss method must be implemented."
        )

    def _compute_repulsive_loss(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _compute_repulsive_loss method must be implemented."
        )

    def _compute_loss(self):
        """Compute the total loss as early_exag * attractive + repulsion_strength * repulsive.

        Subclasses that need a different loss structure (e.g. :class:`TSNEkhorn`)
        can override this method entirely.
        """
        loss = (
            self.early_exaggeration_coeff_ * self._compute_attractive_loss()
            + self.repulsion_strength * self._compute_repulsive_loss()
        )
        return loss

    @torch.no_grad()
    def _compute_gradients(self):
        """Compute gradients directly (used when _use_direct_gradients is True)."""
        gradients = (
            self.early_exaggeration_coeff_ * self._compute_attractive_gradients()
            + self.repulsion_strength * self._compute_repulsive_gradients()
        )
        return gradients

    def _compute_attractive_gradients(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _compute_attractive_gradients method must be implemented "
            "when _use_direct_gradients is True."
        )

    def _compute_repulsive_gradients(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _compute_repulsive_gradients method must be implemented "
            "when _use_direct_gradients is True."
        )

    # --- Early exaggeration ---

    def on_training_step_end(self):
        """End early exaggeration phase when the iteration threshold is reached."""
        if (
            self.early_exaggeration_coeff_ > 1
            and self.n_iter_ == self.early_exaggeration_iter
        ):
            self.early_exaggeration_coeff_ = 1
            # Reinitialize optimizer with post-exaggeration hyperparameters
            # (higher momentum, adjusted learning rate).
            self._set_learning_rate()
            self._configure_optimizer()
            self._configure_scheduler()

        return self

    def _check_n_neighbors(self, n):
        """Validate that the number of samples exceeds perplexity / n_neighbors."""
        for param_name in ("perplexity", "n_neighbors"):
            if hasattr(self, param_name):
                param_value = getattr(self, param_name)
                if n <= param_value:
                    raise ValueError(
                        f"[TorchDR] ERROR : Number of samples is smaller than {param_name} "
                        f"({n} <= {param_value})."
                    )

        return self

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        n_samples = len(X.dataset) if isinstance(X, DataLoader) else X.shape[0]
        self._check_n_neighbors(n_samples)
        # Initialize the mutable exaggeration coefficient (may be reset to 1 during
        # optimization when the early exaggeration phase ends).
        self.early_exaggeration_coeff_ = self.early_exaggeration_coeff

        return super()._fit_transform(X, y)

    # --- Auto learning rate and optimizer ---

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

    def _configure_optimizer(self):
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

    def _configure_scheduler(self):
        if self.early_exaggeration_coeff_ > 1:
            n_iter = min(self.early_exaggeration_iter, self.max_iter)
        else:
            n_iter = self.max_iter - self.early_exaggeration_iter
        super()._configure_scheduler(n_iter)

    # --- Distributed initialization ---

    def _setup_distributed(self, distributed):
        """Configure distributed training state from the ``distributed`` parameter."""
        if distributed == "auto":
            self.distributed = dist.is_initialized()
        else:
            self.distributed = bool(distributed)

        if self.distributed:
            if not dist.is_initialized():
                raise RuntimeError(
                    "[TorchDR] distributed=True requires launching with torchrun. "
                    "Example: torchrun --nproc_per_node=4 your_script.py"
                )

            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_multi_gpu = self.world_size > 1

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            if self.device == "cpu":
                raise ValueError(
                    "[TorchDR] Distributed mode requires GPU (device cannot be 'cpu')"
                )
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.is_multi_gpu = False

    def on_affinity_computation_end(self):
        """Set up chunk_indices_ for the local GPU's portion of the data.

        In distributed mode, the affinity provides chunk bounds (chunk_start_,
        chunk_size_) so each GPU processes a different slice of rows.
        In single-GPU mode, the chunk covers all samples.
        """
        super().on_affinity_computation_end()
        if hasattr(self.affinity_in, "chunk_start_"):
            chunk_start = self.affinity_in.chunk_start_
            chunk_size = self.affinity_in.chunk_size_
        elif self.world_size > 1:
            raise ValueError(
                "[TorchDR] ERROR: Distributed mode is enabled but affinity_in "
                "does not have chunk bounds. Make sure affinity_in has "
                "distributed=True."
            )
        else:
            chunk_start = 0
            chunk_size = self.n_samples_in_

        self.chunk_indices_ = torch.arange(
            chunk_start, chunk_start + chunk_size, device=self.device_
        )

    def _init_embedding(self, X: torch.Tensor):
        """Initialize embedding across ranks (broadcast from rank 0)."""
        # All ranks must run _init_embedding to avoid NCCL deadlocks
        # (e.g., PCA init may trigger distributed ops internally).
        super()._init_embedding(X)

        if self.world_size > 1:
            # Update data in-place to preserve Parameter/ManifoldParameter type.
            if not self.embedding_.data.is_contiguous():
                self.embedding_.data = self.embedding_.data.contiguous()

            dist.broadcast(self.embedding_.data, src=0)

        return self.embedding_


class NegativeSamplingNeighborEmbedding(NeighborEmbedding):
    r"""Neighbor embedding that approximates the repulsive term via negative sampling.

    This class extends :class:`NeighborEmbedding` for methods that
    avoid the :math:`O(n^2)` cost of computing the repulsive term over all
    point pairs. Instead, a fixed number of *negative samples*
    (:attr:`n_negatives`) are drawn uniformly per point at each iteration,
    reducing the repulsive cost to :math:`O(n)`.

    **Negative sampling details:**

    - At each iteration, :attr:`n_negatives` indices are sampled uniformly
      (excluding the point itself) for each point in the local chunk.
    - When :attr:`discard_NNs` is ``True``, nearest neighbors are also
      excluded from the negative samples to avoid conflicting gradients.
    - The sampled indices are stored in :attr:`neg_indices_` and refreshed
      every iteration via :meth:`on_training_step_start`.

    **Inherits** distributed multi-GPU support from
    :class:`NeighborEmbedding`.

    **Subclasses** must implement :meth:`_compute_attractive_loss` and
    :meth:`_compute_repulsive_loss` (or the gradient equivalents).

    **Direct subclasses**: :class:`UMAP`, :class:`LargeVis`,
    :class:`InfoTSNE`, :class:`PACMAP`.

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
        Default is "auto", which corresponds to a linear decay from the learning rate to 0 for `LinearLR`.
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
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
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
    repulsion_strength: float, optional
        Strength of the repulsive term. Default is 1.0.
    n_negatives : int, optional
        Number of negative samples to use. Default is 5.
    check_interval : int, optional
        Number of iterations between two checks for convergence. Default is 50.
    discard_NNs : bool, optional
        Whether to discard nearest neighbors from negative sampling. Default is False.
    compile : bool, default=False
        Whether to use torch.compile for faster computation.
    **kwargs
        All other parameters (including ``distributed``) are forwarded to
        :class:`NeighborEmbedding`.
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
        scheduler_kwargs: Union[Dict, str, None] = "auto",
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        device: str = "auto",
        backend: Union[str, FaissConfig, None] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 1.0,
        early_exaggeration_iter: Optional[int] = None,
        repulsion_strength: float = 1.0,
        n_negatives: int = 5,
        check_interval: int = 50,
        discard_NNs: bool = False,
        compile: bool = False,
        **kwargs,
    ):
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
            repulsion_strength=repulsion_strength,
            check_interval=check_interval,
            compile=compile,
            **kwargs,
        )

        self.n_negatives = n_negatives
        self.discard_NNs = discard_NNs

    def on_affinity_computation_end(self):
        """Build per-row exclusion indices for negative sampling."""
        super().on_affinity_computation_end()

        chunk_size = len(self.chunk_indices_)
        global_self_idx = self.chunk_indices_.unsqueeze(1)

        # Optionally include NN indices (rows aligned with local slice)
        if self.discard_NNs:
            if not hasattr(self, "NN_indices_"):
                self.logger.warning(
                    "NN_indices_ not found. Cannot discard NNs from negative sampling."
                )
                exclude = global_self_idx
            else:
                nn_rows = self.NN_indices_
                if nn_rows.shape[0] != chunk_size:
                    raise ValueError(
                        f"[TorchDR] ERROR: In distributed mode, expected NN_indices_ to have "
                        f"{chunk_size} rows for chunk size, but got {nn_rows.shape[0]}."
                    )
                exclude = torch.cat([global_self_idx, nn_rows], dim=1)
        else:
            exclude = global_self_idx

        # Sort per-row exclusions for searchsorted
        exclude_sorted, _ = exclude.sort(dim=1)
        self.register_buffer(
            "negative_exclusion_indices_", exclude_sorted, persistent=False
        )

        # Safety check on number of available negatives
        n_possible = self.n_samples_in_ - self.negative_exclusion_indices_.shape[1]
        if self.n_negatives > n_possible and self.verbose:
            raise ValueError(
                f"[TorchDR] ERROR : requested {self.n_negatives} negatives but "
                f"only {n_possible} available."
            )

    def on_training_step_start(self):
        """Sample negatives using a unified path for single- and multi-GPU."""
        super().on_training_step_start()

        chunk_size = len(self.chunk_indices_)
        device = self.embedding_.device

        exclusion = self.negative_exclusion_indices_
        excl_width = exclusion.shape[1]

        # Only excluding self-indices
        if excl_width == 1:
            negatives = torch.randint(
                0,
                self.n_samples_in_ - 1,
                (chunk_size, self.n_negatives),
                device=device,
            )
            self_idx = self.chunk_indices_.unsqueeze(1)
            neg_indices = negatives + (negatives >= self_idx).long()

        # Excluding self-indices and NNs indices (computed in on_affinity_computation_end)
        else:
            negatives = torch.randint(
                1,
                self.n_samples_in_ - excl_width,
                (chunk_size, self.n_negatives),
                device=device,
            )
            shifts = torch.searchsorted(exclusion, negatives, right=True)
            neg_indices = negatives + shifts

        self.register_buffer("neg_indices_", neg_indices, persistent=False)
