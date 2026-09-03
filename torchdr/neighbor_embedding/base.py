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
from torchdr.affinity.entropic import _log_Pe
from torchdr.distance import FaissConfig, FaissPlanConfig, pairwise_distances
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.utils import (
    binary_search,
    entropy,
    logsumexp_red,
    to_torch,
    validate_tensor,
)


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
      and :meth:`_compute_repulsive_loss`. When :attr:`_use_closed_form_gradients` is
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
    backend : {"keops", "faiss", None}, FaissConfig, or FaissPlanConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with a low-level expert configuration
        - FaissPlanConfig object: Use FAISS from a high-level execution intent
          (e.g. ``mode="exact"``); the resolved plan is exposed as ``faiss_plan_``
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
        backend: Union[str, FaissConfig, FaissPlanConfig, None] = None,
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
        """Compute gradients directly (used when _use_closed_form_gradients is True)."""
        gradients = (
            self.early_exaggeration_coeff_ * self._compute_attractive_gradients()
            + self.repulsion_strength * self._compute_repulsive_gradients()
        )
        return gradients

    def _compute_attractive_gradients(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _compute_attractive_gradients method must be implemented "
            "when _use_closed_form_gradients is True."
        )

    def _compute_repulsive_gradients(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _compute_repulsive_gradients method must be implemented "
            "when _use_closed_form_gradients is True."
        )

    # --- Input validation and fit ---

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

        # Keep the host offset to avoid synchronizing chunk_indices_ each step.
        self.chunk_start_ = int(chunk_start)
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
    - When :attr:`exclude_neighbors_from_negative_sampling` is ``True``,
      nearest neighbors are also excluded from the negative samples to avoid
      conflicting gradients.
    - The sampled indices are stored in :attr:`neg_indices_` and refreshed
      every iteration via :meth:`on_training_step_start`.

    **Non-parametric transform support:**

    This family also provides the shared machinery for *out-of-sample*
    non-parametric transform. The base implementation handles the generic
    transform lifecycle:

    - find nearest neighbors from new points to the reference training set;
    - build a bipartite affinity from new points to training points;
    - initialize new embeddings from that bipartite graph;
    - optimize only the new points while keeping the fitted training
      embedding frozen.

    The design is intentionally split so the base class owns the generic
    transform pipeline, while each algorithm only provides the
    method-specific bipartite affinity through
    :meth:`_compute_bipartite_affinity`. This keeps the out-of-sample logic
    centralized and prevents each subclass from reimplementing the same
    transform scaffolding.

    **Inherits** distributed multi-GPU support from
    :class:`NeighborEmbedding`.

    **Subclasses** must implement :meth:`_compute_attractive_loss` and
    :meth:`_compute_repulsive_loss` (or the gradient equivalents).
    Subclasses that support non-parametric transform must additionally
    implement :meth:`_compute_bipartite_affinity`.

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
    backend : {"keops", "faiss", None}, FaissConfig, or FaissPlanConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with a low-level expert configuration
        - FaissPlanConfig object: Use FAISS from a high-level execution intent
          (e.g. ``mode="exact"``); the resolved plan is exposed as ``faiss_plan_``
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
    exclude_neighbors_from_negative_sampling : bool, optional
        Whether to exclude nearest neighbors from negative sampling.
        Default is False.
    discard_NNs : bool, optional
        Deprecated alias for ``exclude_neighbors_from_negative_sampling``.
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
        backend: Union[str, FaissConfig, FaissPlanConfig, None] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 1.0,
        early_exaggeration_iter: Optional[int] = None,
        repulsion_strength: float = 1.0,
        n_negatives: int = 5,
        check_interval: int = 50,
        exclude_neighbors_from_negative_sampling: Optional[bool] = None,
        discard_NNs: Optional[bool] = None,
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

        if (
            exclude_neighbors_from_negative_sampling is not None
            and discard_NNs is not None
            and bool(exclude_neighbors_from_negative_sampling) != bool(discard_NNs)
        ):
            raise ValueError(
                "[TorchDR] Conflicting values were provided for "
                "exclude_neighbors_from_negative_sampling and its deprecated "
                "alias discard_NNs."
            )
        if discard_NNs is not None:
            warnings.warn(
                "`discard_NNs` is deprecated; use "
                "`exclude_neighbors_from_negative_sampling` instead.",
                FutureWarning,
                stacklevel=3,
            )

        self.n_negatives = n_negatives
        self.discard_NNs = discard_NNs
        self.exclude_neighbors_from_negative_sampling = bool(
            discard_NNs
            if exclude_neighbors_from_negative_sampling is None
            else exclude_neighbors_from_negative_sampling
        )

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        """Fit and keep a CPU copy of the embedding only when transform is supported.

        The transform path needs access to the fitted reference embedding, but
        storing that extra CPU copy for every estimator would be wasteful.
        The copy is therefore created only for subclasses that opt into the
        non-parametric transform pipeline.
        """
        embedding = super()._fit_transform(X, y)

        if self._supports_non_parametric_transform():
            self.embedding_train_ = embedding.detach().cpu().clone()
        elif hasattr(self, "embedding_train_"):
            delattr(self, "embedding_train_")

        return embedding

    def on_affinity_computation_end(self):
        """Build per-row exclusion indices for negative sampling."""
        super().on_affinity_computation_end()

        chunk_size = len(self.chunk_indices_)
        global_self_idx = self.chunk_indices_.unsqueeze(1)

        # Optionally include NN indices (rows aligned with local slice)
        if self.exclude_neighbors_from_negative_sampling:
            if not hasattr(self, "NN_indices_"):
                self.logger.warning(
                    "NN_indices_ not found. Cannot exclude neighbors from "
                    "negative sampling."
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

        adjusted_exclusion, n_available = self._prepare_exclusion_sampling(
            exclude_sorted, self.n_samples_in_
        )
        self.register_buffer(
            "negative_adjusted_exclusion_", adjusted_exclusion, persistent=False
        )
        self.register_buffer(
            "negative_available_counts_", n_available, persistent=False
        )

    @staticmethod
    def _prepare_exclusion_sampling(exclusion, n_candidates):
        """Prepare row-wise exclusions for uniform compressed-index sampling.

        Invalid padding indices and duplicate exclusions are ignored.  The
        returned adjusted values map a uniform draw in the compressed range
        back into ``[0, n_candidates)`` with ``torch.searchsorted``.
        """
        exclusion = exclusion.long()
        if exclusion.ndim != 2:
            raise ValueError("[TorchDR] exclusion indices must be a 2D tensor.")
        if n_candidates < 1:
            raise ValueError("[TorchDR] negative sampling requires candidates.")

        sentinel = torch.full_like(exclusion, n_candidates)
        valid = (exclusion >= 0) & (exclusion < n_candidates)
        exclusion = torch.where(valid, exclusion, sentinel).sort(dim=1).values
        valid = exclusion < n_candidates

        unique = valid.clone()
        if exclusion.shape[1] > 1:
            unique[:, 1:] &= exclusion[:, 1:] != exclusion[:, :-1]

        n_excluded = unique.sum(dim=1)
        n_available = n_candidates - n_excluded
        if (n_available <= 0).any():
            raise ValueError(
                "[TorchDR] No candidates remain after applying negative-sampling "
                "exclusions."
            )

        width = exclusion.shape[1]
        columns = torch.arange(width, device=exclusion.device, dtype=torch.long)
        compact = (n_candidates + columns).unsqueeze(0).expand_as(exclusion).clone()
        compact_positions = unique.cumsum(dim=1) - 1
        rows = torch.arange(exclusion.shape[0], device=exclusion.device)
        rows = rows.unsqueeze(1).expand_as(exclusion)
        compact[rows[unique], compact_positions[unique]] = exclusion[unique]

        # For the j-th sorted exclusion e_j, compare compressed indices against
        # e_j - j.  Sentinel entries become n_candidates and remain at the end.
        adjusted_exclusion = (compact - columns).contiguous()
        return adjusted_exclusion, n_available.long()

    @staticmethod
    def _draw_with_exclusions(adjusted_exclusion, n_available, n_draws, offset=0):
        """Draw uniformly with replacement from row-wise allowed indices."""
        compressed = (
            torch.rand(
                (adjusted_exclusion.shape[0], n_draws),
                device=adjusted_exclusion.device,
            )
            * n_available.unsqueeze(1)
        ).long()
        shifts = torch.searchsorted(adjusted_exclusion, compressed, right=True)
        return compressed + shifts + offset

    def on_training_step_start(self):
        """Sample negatives using a unified path for single- and multi-GPU."""
        super().on_training_step_start()

        neg_indices = self._draw_with_exclusions(
            self.negative_adjusted_exclusion_,
            self.negative_available_counts_,
            self.n_negatives,
        )

        self.register_buffer("neg_indices_", neg_indices, persistent=False)

    # --- Non-parametric transform ---

    def _get_n_neighbors_transform(self, n_train):
        """Return a valid support size for the transform kNN search.

        UMAP uses its configured neighborhood size.  Entropic methods need a
        support larger than their effective perplexity; this mirrors the
        ``3 * perplexity`` sparse support used by :class:`EntropicAffinity`.
        """
        if n_train < 3:
            raise ValueError(
                "[TorchDR] At least 3 training samples are required for transform."
            )
        if hasattr(self, "n_neighbors"):
            requested = int(self.n_neighbors)
        elif hasattr(self, "perplexity"):
            requested = int(3 * self.perplexity)
        else:
            raise ValueError("[TorchDR] Cannot determine n_neighbors for transform.")

        # The dense distance helper reserves k >= n_train for a full matrix and
        # does not return indices, so keep a proper k-NN result on every backend.
        return max(2, min(requested, n_train - 1))

    def _compute_bipartite_entropic_affinity(self, C):
        """Build a fit-scaled entropic affinity on a bipartite distance graph."""
        support_size = C.shape[1]
        if self.perplexity > support_size:
            raise ValueError(
                f"[TorchDR] Transform perplexity ({self.perplexity}) exceeds "
                f"the available neighbor support ({support_size}). Use a larger "
                "training reference set or a smaller perplexity."
            )
        if self.perplexity == support_size:
            # The only distribution attaining maximal entropy log(k) is the
            # uniform distribution. Avoid sending this boundary root toward an
            # effectively infinite bandwidth.
            return torch.full_like(C, 1.0 / (C.shape[0] * support_size))

        perplexity = torch.tensor(self.perplexity, dtype=C.dtype, device=C.device)
        target_entropy = perplexity.log() + 1

        def entropy_gap(eps):
            log_P = _log_Pe(C, eps)
            log_P_normalized = log_P - logsumexp_red(log_P, dim=1)
            return entropy(log_P_normalized, log=True).reshape(-1) - target_entropy

        eps = binary_search(
            f=entropy_gap,
            n=C.shape[0],
            max_iter=self.max_iter_affinity,
            dtype=C.dtype,
            device=C.device,
        )

        log_P = _log_Pe(C, eps)
        log_P -= logsumexp_red(log_P, dim=1)

        # Fit-time EntropicAffinity gives every row mass 1 / n so the complete
        # attractive distribution has unit mass.  Preserve that invariant for
        # the n_new bipartite query rows.
        log_P -= torch.log(torch.tensor(C.shape[0], dtype=C.dtype, device=C.device))
        return log_P.exp()

    def _compute_bipartite_affinity(self, C, indices):
        """Compute bipartite affinity from new points to training points.

        This hook is the method-specific part of the shared non-parametric
        transform pipeline. The base class handles neighbor search,
        initialization, and optimization; subclasses only need to define how
        distances from new points to training neighbors are converted into
        affinity weights.

        Parameters
        ----------
        C : torch.Tensor of shape (n_new, k)
            Distances from new points to their k nearest training neighbors.
        indices : torch.Tensor of shape (n_new, k)
            Indices of k nearest training neighbors.

        Returns
        -------
        affinity : torch.Tensor of shape (n_new, k)
            Bipartite affinity weights (non-negative, not symmetrized).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "_compute_bipartite_affinity for transform."
        )

    def _supports_non_parametric_transform(self):
        """Return whether the subclass opted into non-parametric transform.

        Support is explicit: a subclass enables the shared transform pipeline
        by overriding :meth:`_compute_bipartite_affinity`.
        """
        return (
            self.__class__._compute_bipartite_affinity
            is not NegativeSamplingNeighborEmbedding._compute_bipartite_affinity
        )

    def _get_fit_learning_rate(self):
        """Return the learning rate configured at the start of fit."""
        saved_lr = getattr(self, "lr_", None)
        had_lr = hasattr(self, "lr_")
        saved_early_exaggeration = getattr(self, "early_exaggeration_coeff_", None)
        had_early_exaggeration = hasattr(self, "early_exaggeration_coeff_")

        try:
            self.early_exaggeration_coeff_ = self.early_exaggeration_coeff
            self._set_learning_rate()
            return float(self.lr_)
        finally:
            if had_lr:
                self.lr_ = saved_lr
            elif hasattr(self, "lr_"):
                delattr(self, "lr_")

            if had_early_exaggeration:
                self.early_exaggeration_coeff_ = saved_early_exaggeration
            elif hasattr(self, "early_exaggeration_coeff_"):
                delattr(self, "early_exaggeration_coeff_")

    def _get_transform_learning_rate(self):
        """Return the transform learning rate as 1/4 of the fit-time LR."""
        return self._get_fit_learning_rate() / 4.0

    def _get_max_iter_transform(self):
        """Return the number of optimization steps used during transform."""
        return min(self.max_iter // 3, 100)

    def _sample_transform_neg_indices(self, n_new, n_train, nn_indices):
        """Sample transform negatives from the frozen training embedding.

        During non-parametric transform, repulsive negatives are drawn only
        from the reference training points. When
        :attr:`exclude_neighbors_from_negative_sampling` is enabled, the
        positive training neighbors of each new point are removed from that
        negative pool.
        """
        if not self.exclude_neighbors_from_negative_sampling:
            return torch.randint(
                n_new,
                n_new + n_train,
                (n_new, self.n_negatives),
                device=self.device_,
            )

        adjusted_exclusion, n_available = self._prepare_exclusion_sampling(
            nn_indices, n_train
        )
        return self._draw_with_exclusions(
            adjusted_exclusion, n_available, self.n_negatives, offset=n_new
        )

    def _initialize_transform_embedding(
        self, affinity, nn_indices, train_emb, neighbor_distances=None
    ):
        """Initialize transformed points from the bipartite neighbor graph.

        The default initialization is the affinity-weighted average of the
        training neighbors' fitted embeddings. Subclasses can override this to
        match algorithm-specific initialization rules while still reusing the
        shared transform pipeline.
        """
        weights = affinity / affinity.sum(dim=1, keepdim=True).clamp(min=1e-10)
        neighbor_emb = train_emb[nn_indices.long()]
        return (weights.unsqueeze(-1) * neighbor_emb).sum(dim=1)

    def _transform(self, X_new, X_train=None):
        """Transform new data using non-parametric neighbor embedding.

        Finds nearest neighbors in the training data, builds a bipartite
        affinity graph, initializes positions from that graph, and optimizes
        only the new points while keeping the fitted training embedding
        frozen.

        Parameters
        ----------
        X_new : array-like of shape (n_new, n_features)
            New data to transform.
        X_train : array-like of shape (n_train, n_features)
            Training data used during fit, with the same samples and row order.
            Required because training features are not stored to avoid memory
            overhead.

        Returns
        -------
        embedding_new : torch.Tensor of shape (n_new, n_components)
            Embedding of the new data.
        """
        if not self._supports_non_parametric_transform():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support non-parametric transform."
            )

        if X_train is None:
            raise ValueError(
                "[TorchDR] X_train is required for non-parametric transform. "
                "Pass the training data: model.transform(X_new, X_train=X_train)"
            )

        if not hasattr(self, "embedding_train_"):
            raise RuntimeError(
                "[TorchDR] Training embedding not available. "
                "Call fit() or fit_transform() first."
            )

        X_new = validate_tensor(to_torch(X_new))
        X_train = validate_tensor(to_torch(X_train))

        if X_new.shape[1] != self.n_features_in_:
            raise ValueError(
                f"[TorchDR] X_new has {X_new.shape[1]} features, but the model "
                f"was fitted with {self.n_features_in_}."
            )
        if X_train.shape[1] != self.n_features_in_:
            raise ValueError(
                f"[TorchDR] X_train has {X_train.shape[1]} features, but the model "
                f"was fitted with {self.n_features_in_}."
            )
        if X_train.shape[0] != self.embedding_train_.shape[0]:
            raise ValueError(
                f"[TorchDR] X_train has {X_train.shape[0]} samples, but the fitted "
                f"reference embedding has {self.embedding_train_.shape[0]}. Pass "
                "the same training samples, in the same order, that were used in fit."
            )

        compute_dtype = self.embedding_train_.dtype
        X_new = X_new.to(device=self.device_, dtype=compute_dtype)
        X_train = X_train.to(device=self.device_, dtype=compute_dtype)

        # Step 1: kNN from new points to training points
        k = self._get_n_neighbors_transform(X_train.shape[0])
        C, nn_indices = pairwise_distances(
            X=X_new,
            Y=X_train,
            metric=self.metric,
            backend=self.backend,
            k=k,
            return_indices=True,
            device=self.device_,
        )
        if self.metric in {"euclidean", "sqeuclidean", "manhattan"}:
            # Round-off in the quadratic squared-distance formula can produce
            # tiny negative values, including for exact cross-set matches.
            C.clamp_min_(0)

        # Step 2: bipartite affinity (subclass-specific)
        affinity = self._compute_bipartite_affinity(C, nn_indices)

        train_emb = self.embedding_train_.to(device=self.device_)

        # Step 3: initialize from the bipartite graph
        embedding_new = self._initialize_transform_embedding(
            affinity, nn_indices, train_emb, neighbor_distances=C
        )

        # Step 4: optimize with frozen training embeddings
        embedding_new = self._optimize_transform(
            embedding_new, affinity, nn_indices, train_emb
        )
        return embedding_new

    def _enter_transform(self, embedding_new, train_emb, affinity, nn_indices):
        """Save fit-time state and set up for transform.

        Builds a combined embedding ``[embedding_new, train_emb]`` so that
        the existing ``_compute_loss`` / ``_compute_gradients`` methods
        work unmodified — queries index into the new part and keys index
        into the training part. This is the key design choice that keeps the
        transform code small: the transform path reuses the usual objective
        implementation instead of introducing a second optimization codepath.

        Subclasses can override to set up additional state (e.g. UMAP's
        edge-sampling buffers). Must call ``super()._enter_transform(...)``.

        Returns
        -------
        saved : dict
            State to restore in :meth:`_exit_transform`.
        """
        n_new = embedding_new.shape[0]

        saved = {}
        for attr in (
            "embedding_",
            "chunk_indices_",
            "chunk_start_",
            "NN_indices_",
            "affinity_in_",
            "n_samples_in_",
            "early_exaggeration_coeff_",
            "n_iter_",
            "neg_indices_",
        ):
            saved[attr] = (hasattr(self, attr), getattr(self, attr, None))

        chunk_indices = torch.arange(n_new, device=self.device_)
        transform_nn_indices = nn_indices + n_new
        transform_n_iter = torch.tensor(0, device=self.device_)

        try:
            # Most embeddings are leaf tensors, but hyperbolic initialization
            # uses an nn.Parameter. Remove its registration before assigning a
            # temporary concatenated tensor in the optimization loop.
            if isinstance(self.embedding_, torch.nn.Parameter):
                delattr(self, "embedding_")

            self.chunk_indices_ = chunk_indices
            self.chunk_start_ = 0
            self.NN_indices_ = transform_nn_indices
            self.affinity_in_ = affinity
            # Existing objectives normalize over query rows. The concatenated
            # embedding contains reference rows for indexing, but only new rows
            # are optimized and must contribute to this normalization.
            self.n_samples_in_ = n_new
            self.early_exaggeration_coeff_ = 1
            self.n_iter_ = transform_n_iter
        except Exception:
            self._exit_transform(saved)
            raise

        return saved

    def _exit_transform(self, saved):
        """Restore fit-time state after transform."""
        for attr, (existed, value) in saved.items():
            if existed:
                setattr(self, attr, value)
            elif hasattr(self, attr):
                delattr(self, attr)

    def _optimize_transform(self, embedding_new, affinity, nn_indices, train_emb):
        """Optimize new embeddings with frozen training embeddings via SGD.

        Uses the concatenation trick: builds
        ``embedding_ = cat([embedding_new, train_emb])`` so that the
        existing ``_compute_loss`` / ``_compute_gradients`` methods
        can be reused without modification. Only ``embedding_new`` is a trainable
        parameter; ``train_emb`` acts as the frozen reference geometry.

        Parameters
        ----------
        embedding_new : torch.Tensor of shape (n_new, n_components)
            Initial positions for new points.
        affinity : torch.Tensor of shape (n_new, k)
            Bipartite affinity from new to training points.
        nn_indices : torch.Tensor of shape (n_new, k)
            Indices of nearest training neighbors.
        train_emb : torch.Tensor of shape (n_train, n_components)
            Frozen training embeddings.

        Returns
        -------
        embedding_new : torch.Tensor of shape (n_new, n_components)
            Optimized positions.
        """
        n_new = embedding_new.shape[0]
        n_train = train_emb.shape[0]
        embedding_new = torch.nn.Parameter(embedding_new.clone())

        # LR: 1/4 of fit-time LR (following umap-learn)
        lr = self._get_transform_learning_rate()
        max_iter_transform = self._get_max_iter_transform()

        optimizer = torch.optim.SGD([embedding_new], lr=lr)

        saved = None
        try:
            saved = self._enter_transform(
                embedding_new, train_emb, affinity, nn_indices
            )
            for step in range(max_iter_transform):
                # Match the linear learning-rate decay used by UMAP's transform
                # and by the default fit schedulers of these methods.
                optimizer.param_groups[0]["lr"] = lr * (
                    1.0 - step / max(max_iter_transform, 1)
                )

                # Rebuild combined embedding each step (new points change)
                self.embedding_ = torch.cat([embedding_new, train_emb.detach()], dim=0)
                self.n_iter_.fill_(step)

                self.neg_indices_ = self._sample_transform_neg_indices(
                    n_new, n_train, nn_indices
                )

                optimizer.zero_grad(set_to_none=True)

                if getattr(self, "_use_closed_form_gradients", False):
                    gradients = self._compute_gradients()
                    embedding_new.grad = gradients
                else:
                    loss = self._compute_loss()
                    loss.backward()

                optimizer.step()
        finally:
            if saved is not None:
                self._exit_transform(saved)

        return embedding_new.detach()
