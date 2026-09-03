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
from torchdr.utils import binary_search

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
    discard_NNs : bool, optional
        Deprecated alias for ``exclude_neighbors_from_negative_sampling``.
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
        exclude_neighbors_from_negative_sampling: Optional[bool] = None,
        discard_NNs: Optional[bool] = None,
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
            discard_NNs=discard_NNs,
            compile=compile,
            n_negatives=self.n_negatives,
            distributed=distributed,
            **kwargs,
        )

    @staticmethod
    def _flatten_padded_edges(nn_indices):
        """Flatten a max-degree-padded neighbor grid to flat per-edge arrays.

        The symmetrized UMAP affinity is delivered as a ``(chunk, max_degree)``
        grid padded with ``-1`` column indices, where ``max_degree`` can be far
        larger than the mean degree. This returns, in row-major order over the
        real (non-padded) edges, the local ``source`` row and the global
        ``target`` column of every edge, together with the boolean ``mask`` so
        that aligned per-edge tensors (e.g. the affinity values) can be
        flattened the same way.
        """
        mask = nn_indices >= 0
        counts = mask.sum(dim=1)
        source = torch.repeat_interleave(
            torch.arange(nn_indices.shape[0], device=nn_indices.device), counts
        )
        target = nn_indices[mask].contiguous()
        return source, target, mask

    def on_affinity_computation_end(self):
        super().on_affinity_computation_end()

        # Flatten the max-degree-padded affinity grid to its real edges only
        # (a CSR-style layout), so the closed-form attractive gradient runs over
        # ``nnz`` edges instead of ``chunk * max_degree`` mostly-padded slots.
        source, target, edge_mask = self._flatten_padded_edges(self.NN_indices_)
        edge_affinity = self.affinity_in_[edge_mask]

        # Remove small affinity edges (padded slots are already excluded).
        A_max = edge_affinity.max()
        threshold = A_max / self.max_iter
        small_affinity_edges = edge_affinity <= threshold

        if self.verbose:
            kept_pct = (~small_affinity_edges).float().mean().item() * 100
            self.logger.info(f"Keeping {kept_pct:.1f}% of affinity edges.")

        epochs_per_sample = edge_affinity.add(1e-3).reciprocal_().mul_(A_max)
        epochs_per_sample.masked_fill_(
            small_affinity_edges, float("inf")
        )  # avoid updating these edges

        self.register_buffer("attractive_source_", source, persistent=False)
        self.register_buffer("attractive_target_", target, persistent=False)
        self.register_buffer("epochs_per_sample", epochs_per_sample, persistent=False)
        self.register_buffer(
            "epoch_of_next_sample", epochs_per_sample.clone(), persistent=False
        )

        # The padded grid is no longer needed: UMAP uses closed-form gradients
        # (no loss reads ``affinity_in_``) and both gradient terms now index the
        # flat edge buffers. Free it to reclaim the max-degree padding overhead.
        del self.affinity_in_
        del self.NN_indices_

    def _compute_attractive_gradients(self):
        source = self.attractive_source_
        diff = (
            self.embedding_[self.chunk_indices_[source]]
            - self.embedding_[self.attractive_target_]
        )
        D = diff.pow(2).sum(dim=1)
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

        # Segment-reduce the per-edge contributions back to their source rows.
        grad = torch.zeros(
            (self.chunk_indices_.shape[0], self.embedding_.shape[1]),
            dtype=self.embedding_.dtype,
            device=self.embedding_.device,
        )
        grad.index_add_(0, source, diff.mul_(D.unsqueeze(1)))
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
        # mask_affinity_in_ is a flat per-edge mask, so the per-row count of
        # active positive edges is a segment sum over their source rows.
        active_positive = torch.zeros(
            self.chunk_indices_.shape[0],
            dtype=torch.long,
            device=self.embedding_.device,
        )
        active_positive.index_add_(
            0, self.attractive_source_, self.mask_affinity_in_.to(torch.long)
        )
        neg_counts = (active_positive * self.negative_sample_rate).to(torch.long)
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
        # umap-learn reduces local_connectivity by one for transform.  With
        # UMAP's default local_connectivity=1 this means rho=0, unlike fit-time
        # graph construction.  Taking the nearest distance here would force one
        # affinity to 1 for every query and make every point look like an exact
        # match during initialization.
        rho = torch.zeros(C.shape[0], dtype=C.dtype, device=C.device)

        log_n_neighbors = torch.log2(
            torch.tensor(self.n_neighbors, dtype=C.dtype, device=C.device)
        )

        def marginal_gap(eps):
            # Match smooth_knn_dist's bipartite transform convention: the
            # closest edge participates in the graph but is omitted from the
            # bandwidth calibration sum.
            log_marg = _log_P_UMAP(C[:, 1:], rho, eps).logsumexp(1)
            return log_marg.exp().reshape(-1) - log_n_neighbors

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

    def _initialize_transform_embedding(
        self, affinity, nn_indices, train_emb, neighbor_distances=None
    ):
        """Match UMAP's transform initialization when exact matches exist.

        The default weighted-average initialization from the base class is kept,
        except that rows containing a zero-distance neighbor are snapped to the
        corresponding training embedding exactly, as in ``umap-learn``.
        """
        embedding_new = super()._initialize_transform_embedding(
            affinity,
            nn_indices,
            train_emb,
            neighbor_distances=neighbor_distances,
        )
        if neighbor_distances is None:
            return embedding_new

        exact_match = neighbor_distances == 0
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
        epochs_per_sample = self._make_transform_epochs_per_sample(
            affinity, self._get_max_iter_transform()
        )
        saved = super()._enter_transform(embedding_new, train_emb, affinity, nn_indices)

        # Save UMAP-specific state
        for attr in (
            "epochs_per_sample",
            "epoch_of_next_sample",
            "mask_affinity_in_",
            "attractive_source_",
            "attractive_target_",
        ):
            saved[attr] = (hasattr(self, attr), getattr(self, attr, None))

        # Flatten the (offset) transform neighbor grid to the same flat edge
        # layout used at fit time so the shared closed-form gradient runs over
        # real edges. super()._enter_transform set NN_indices_ to the global
        # transform neighbor indices.
        source, target, edge_mask = self._flatten_padded_edges(self.NN_indices_)
        self.attractive_source_ = source
        self.attractive_target_ = target
        self.epochs_per_sample = epochs_per_sample[edge_mask]
        self.epoch_of_next_sample = self.epochs_per_sample.clone()

        return saved
