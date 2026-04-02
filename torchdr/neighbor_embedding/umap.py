"""UMAP algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Union, Type
import torch
import numpy as np

from torchdr.affinity import UMAPAffinity
from torchdr.affinity.knn_normalized import _log_P_UMAP
from torchdr.neighbor_embedding.base import NegativeSamplingNeighborEmbedding
from torchdr.distance import pairwise_distances_indexed, FaissConfig
from torchdr.utils import binary_search, kmin

from scipy.optimize import curve_fit


# from umap/umap/umap_.py
def find_ab_params(spread, min_dist):
    """Fit a, b params as in UMAP.

    Fit (a, b) for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0].item(), params[1].item()


class UMAP(NegativeSamplingNeighborEmbedding):
    r"""UMAP introduced in :cite:`mcinnes2018umap` and further studied in :cite:`damrich2021umap`.

    It uses a :class:`~torchdr.UMAPAffinity` as input affinity :math:`\mathbf{P}`
    and output affinity :math:`Q_{ij} = (1 + a \| \mathbf{z}_i - \mathbf{z}_j \|^{2b})^{-1}` where :math:`a, b` are fitted from ``min_dist`` and ``spread``.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \sum_{i,j \in \mathrm{Neg}(i)} \log (1 - Q_{ij})

    where :math:`\mathrm{Neg}(i)` is the set of negatives samples for point :math:`i`.

    Note
    ----
    This implementation supports multi-GPU training when launched with ``torchrun``.
    Set ``distributed='auto'`` (default) to automatically detect and use multiple GPUs.
    It also supports the shared non-parametric transform path implemented in
    :class:`NegativeSamplingNeighborEmbedding`.

    Parameters
    ----------
    n_neighbors : float, optional
        Number of nearest neighbors.
    n_components : int, optional
        Dimension of the embedding space.
    min_dist : float, optional
        Minimum distance between points in the embedding space.
    spread : float, optional
        The effective scale of the embedded points. Used to configure the UMAPAffinityOut.
    a : float, optional
        Parameter for the Student t-distribution.
    b : float, optional
        Parameter for the Student t-distribution.
    lr : float, optional
        Learning rate for the algorithm, by default 1e-1.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto'.
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is "LinearLR".
    scheduler_kwargs : dict, 'auto', or None, optional
        Additional keyword arguments for the scheduler. Default is 'auto', which
        corresponds to a linear decay from the learning rate to 0 for `LinearLR`.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    min_grad_norm : float, optional
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm. by default 2000.
    device : str, optional
        Device to use, by default "auto".
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
        Default is "faiss".
    verbose : bool, optional
        Verbosity, by default False.
    random_state : float, optional
        Random seed for reproducibility, by default None.
    max_iter_affinity : int, optional
        Number of maximum iterations for the input affinity computation.
    metric : {'euclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    n_negatives : int, optional
        Number of negative samples for the noise-contrastive loss, by default 10.
    check_interval : int, optional
        Check interval for the algorithm, by default 50.
    exclude_neighbors_from_negative_sampling : bool, optional
        Whether to exclude nearest neighbors from negative sampling.
        Default is False.
    compile : bool, optional
        Whether to compile the algorithm using torch.compile. Default is False.
    distributed : bool or 'auto', optional
        Whether to use distributed computation across multiple GPUs.
        - "auto": Automatically detect if running with torchrun (default)
        - True: Force distributed mode (requires torchrun)
        - False: Disable distributed mode
        Default is "auto".
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 30,
        n_components: int = 2,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        lr: float = 1e0,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = None,
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = "LinearLR",
        scheduler_kwargs: Union[Dict, str, None] = "auto",
        init: str = "pca",
        init_scaling: float = 1e-4,
        min_grad_norm: float = 1e-7,
        max_iter: int = 1000,
        device: str = "auto",
        backend: Union[str, FaissConfig, None] = "faiss",
        verbose: bool = False,
        random_state: Optional[float] = None,
        max_iter_affinity: int = 100,
        metric: str = "sqeuclidean",
        negative_sample_rate: int = 5,
        check_interval: int = 50,
        exclude_neighbors_from_negative_sampling: bool = False,
        compile: bool = False,
        distributed: Union[bool, str] = "auto",
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.metric = metric
        self.max_iter_affinity = max_iter_affinity
        self.negative_sample_rate = negative_sample_rate

        self.sparsity = True
        self._use_closed_form_gradients = True
        self._eps = 1e-3

        if a is None or b is None:
            a, b = find_ab_params(self.spread, self.min_dist)
        self._a = a
        self._b = b

        self.n_negatives = int(self.negative_sample_rate * self.n_neighbors)

        affinity_in = UMAPAffinity(
            n_neighbors=n_neighbors,
            metric=metric,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=self.sparsity,
            compile=compile,
            distributed=distributed,
        )

        super().__init__(
            affinity_in=affinity_in,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            check_interval=check_interval,
            exclude_neighbors_from_negative_sampling=exclude_neighbors_from_negative_sampling,
            compile=compile,
            n_negatives=self.n_negatives,
            distributed=distributed,
            **kwargs,
        )

    def on_affinity_computation_end(self):
        super().on_affinity_computation_end()

        # Remove small affinity edges
        A_max = self.affinity_in_.max()
        threshold = A_max / self.max_iter
        small_affinity_edges = self.affinity_in_ <= threshold

        if self.verbose:
            kept_pct = (~small_affinity_edges).float().mean().item() * 100
            self.logger.info(f"Keeping {kept_pct:.1f}% of affinity edges.")

        self.affinity_in_.add_(1e-3).reciprocal_().mul_(A_max)
        self.affinity_in_.masked_fill_(
            small_affinity_edges, float("inf")
        )  # avoid updating these edges
        self.register_buffer("epochs_per_sample", self.affinity_in_, persistent=False)
        self.register_buffer(
            "epoch_of_next_sample", self.epochs_per_sample.clone(), persistent=False
        )

    def _compute_attractive_gradients(self):
        D = pairwise_distances_indexed(
            self.embedding_,
            query_indices=self.chunk_indices_,
            key_indices=self.NN_indices_,
            metric="sqeuclidean",
        )
        positive_edges = D > 0
        D_ = 1 + self._a * D**self._b
        D.pow_(self._b - 1)
        D.mul_(2 * self._a * self._b).div_(D_)
        D.masked_fill_(~positive_edges, 0)  # prevent infinities when b < 1

        # UMAP keeps a per-edge counter (epoch_of_next_sample) so that stronger edges
        # (higher affinity → smaller epochs_per_sample) get updated more often.
        mask_affinity_in = self.epoch_of_next_sample <= self.n_iter_ + 1
        self.register_buffer("mask_affinity_in_", mask_affinity_in, persistent=False)
        self.epoch_of_next_sample[self.mask_affinity_in_] += self.epochs_per_sample[
            self.mask_affinity_in_
        ]
        D.masked_fill_(~self.mask_affinity_in_, 0)

        diff = (
            self.embedding_[self.chunk_indices_].unsqueeze(1)
            - self.embedding_[self.NN_indices_]
        )
        grad = torch.einsum("ijk,ij->ik", diff, D)
        grad.clamp_(-4, 4)  # clamp as in umap repo
        return grad

    def _compute_repulsive_gradients(self):
        D = pairwise_distances_indexed(
            self.embedding_,
            query_indices=self.chunk_indices_,
            key_indices=self.neg_indices_,
            metric="sqeuclidean",
        )
        D_ = 1 + self._a * D**self._b
        D.add_(self._eps)
        D.mul_(D_)
        D.reciprocal_().mul_(-2 * self._b)

        # Filter to keep 'negative_sample_rate' negative edges per positive edge.
        neg_counts = (self.mask_affinity_in_.sum(dim=1) * self.negative_sample_rate).to(
            torch.long
        )
        col_idx = torch.arange(self.n_negatives, device=self.embedding_.device)
        filtered_edges = col_idx[None, :].ge(neg_counts[:, None])
        D.masked_fill_(filtered_edges, 0)

        diff = (
            self.embedding_[self.chunk_indices_].unsqueeze(1)
            - self.embedding_[self.neg_indices_]
        )
        grad = torch.einsum("ijk,ij->ik", diff, D)
        grad.clamp_(-4, 4)  # clamp as in umap repo
        return grad

    # --- Non-parametric transform ---

    def _compute_bipartite_affinity(self, C, indices):
        """Build the UMAP bipartite affinity used during transform.

        This is the UMAP-specific hook for the shared non-parametric transform
        pipeline in :class:`NegativeSamplingNeighborEmbedding`. It mirrors the
        unsymmetrized UMAP neighbor graph construction on the bipartite graph
        from new points to the fitted training set.
        """
        rho = kmin(C, k=1, dim=1)[0].squeeze(-1).contiguous()

        log_n_neighbors = torch.log2(
            torch.tensor(self.n_neighbors, dtype=C.dtype, device=C.device)
        )

        def marginal_gap(eps):
            log_marg = _log_P_UMAP(C, rho, eps).logsumexp(1)
            return log_marg.exp().squeeze() - log_n_neighbors

        eps = binary_search(
            f=marginal_gap,
            n=C.shape[0],
            max_iter=self.max_iter_affinity,
            dtype=C.dtype,
            device=C.device,
        )

        return _log_P_UMAP(C, rho, eps).exp()

    def _make_transform_epochs_per_sample(self, affinity, n_epochs):
        """Convert transform edge strengths into UMAP's epoch schedule.

        This keeps the transform path aligned with UMAP's usual edge-sampling
        logic while still using TorchDR's vectorized, mask-based optimizer.
        """
        epochs_per_sample = torch.full_like(affinity, float("inf"))
        if n_epochs <= 0:
            return epochs_per_sample

        max_affinity = affinity.max()
        if max_affinity <= 0:
            return epochs_per_sample

        threshold = max_affinity / float(n_epochs)
        active_edges = affinity >= threshold
        eps = torch.finfo(affinity.dtype).tiny
        epochs_per_sample[active_edges] = max_affinity / affinity[active_edges].clamp(
            min=eps
        )
        return epochs_per_sample

    def _initialize_transform_embedding(self, affinity, nn_indices, train_emb):
        """Match UMAP's transform initialization when exact matches exist.

        The default weighted-average initialization from the base class is kept,
        except that rows with an affinity of 1 are snapped to the corresponding
        training embedding exactly, as in ``umap-learn``.
        """
        embedding_new = super()._initialize_transform_embedding(
            affinity, nn_indices, train_emb
        )
        exact_match = torch.isclose(
            affinity, torch.ones_like(affinity), atol=1e-6, rtol=0.0
        )
        if exact_match.any():
            exact_rows = exact_match.any(dim=1)
            exact_cols = exact_match.to(torch.int64).argmax(dim=1)
            embedding_new[exact_rows] = train_emb[
                nn_indices[exact_rows, exact_cols[exact_rows]].long()
            ]
        return embedding_new

    def _enter_transform(self, embedding_new, train_emb, affinity, nn_indices):
        """Set up UMAP edge-sampling state for transform.

        Reuses the same edge-sampling schedule as fit, but on the bipartite
        graph between new points and the frozen training embedding. The actual
        optimization still runs through TorchDR's vectorized mask-based update
        path rather than ``umap-learn``'s edge-wise CPU loop.
        """
        saved = super()._enter_transform(embedding_new, train_emb, affinity, nn_indices)

        # Save UMAP-specific state
        for attr in ("epochs_per_sample", "epoch_of_next_sample", "mask_affinity_in_"):
            saved[attr] = getattr(self, attr, None)

        epochs_per_sample = self._make_transform_epochs_per_sample(
            affinity, self._get_max_iter_transform()
        )
        self.epochs_per_sample = epochs_per_sample
        self.epoch_of_next_sample = epochs_per_sample.clone()

        return saved
