"""UMAP algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Union, Type
import torch
import numpy as np

from torchdr.affinity import UMAPAffinity
from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from torchdr.distance import symmetric_pairwise_distances_indices

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


class UMAP(SampledNeighborEmbedding):
    r"""UMAP introduced in :cite:`mcinnes2018umap` and further studied in :cite:`damrich2021umap`.

    It uses a :class:`~torchdr.UMAPAffinity` as input
    affinity :math:`\mathbf{P}`.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \sum_{i,j \in \mathrm{Neg}(i)} \log (1 - Q_{ij})

    where :math:`\mathrm{Neg}(i)` is the set of negatives samples for point :math:`i`.

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
    init : {'normal', 'pca', 'umap_spectral'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'. 'umap_spectral' uses the
        original UMAP package's spectral initialization.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    min_grad_norm : float, optional
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm. by default 2000.
    device : str, optional
        Device to use, by default "auto".
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is "faiss".
    verbose : bool, optional
        Verbosity, by default False.
    random_state : float, optional
        Random seed for reproducibility, by default None.
    tol_affinity : float, optional
        Precision threshold for the input affinity computation.
    max_iter_affinity : int, optional
        Number of maximum iterations for the input affinity computation.
    metric_in : {'euclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'euclidean'.
    metric_out : {'euclidean', 'manhattan'}, optional
        Metric to use for the output affinity, by default 'euclidean'.
    n_negatives : int, optional
        Number of negative samples for the noise-contrastive loss, by default 10.
    check_interval : int, optional
        Check interval for the algorithm, by default 50.
    discard_NNs : bool, optional
        Whether to discard the nearest neighbors from the negative sampling.
        Default is False.
    compile : bool, optional
        Whether to compile the algorithm using torch.compile. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 30,
        n_components: int = 2,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        lr: float = 1e-1,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = None,
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = "LinearLR",
        scheduler_kwargs: Union[Dict, str, None] = "auto",
        init: str = "pca",
        init_scaling: float = 1e-4,
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        device: Optional[str] = None,
        backend: Optional[str] = "faiss",
        verbose: bool = False,
        random_state: Optional[float] = None,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "sqeuclidean",
        negative_sample_rate: int = 5,
        check_interval: int = 50,
        discard_NNs: bool = False,
        compile: bool = False,
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.metric_in = metric_in
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity
        self.negative_sample_rate = negative_sample_rate

        self.metric_out = "sqeuclidean"
        self.sparsity = True
        self._use_direct_gradients = True
        self._eps = 1e-3

        if a is None or b is None:
            a, b = find_ab_params(self.spread, self.min_dist)
        self._a = a
        self._b = b

        affinity_in = UMAPAffinity(
            n_neighbors=n_neighbors,
            metric=metric_in,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=self.sparsity,
            compile=compile,
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
            discard_NNs=discard_NNs,
            compile=compile,
            n_negatives=int(self.negative_sample_rate * self.n_neighbors),
            **kwargs,
        )

    # def _compute_repulsive_loss(self):
    #     D = symmetric_pairwise_distances_indices(
    #         self.embedding_,
    #         metric=self.metric_out,
    #         indices=self.neg_indices_,
    #         compile=self.compile,
    #     )[0]
    #     D = self._a * D**self._b
    #     D = 1 / (2 + D)  # sigmoid trick to avoid numerical instability
    #     return -sum_red((1 - D).log(), dim=(0, 1))

    # def _compute_attractive_loss(self):
    #     D = symmetric_pairwise_distances_indices(
    #         self.embedding_,
    #         metric=self.metric_out,
    #         indices=self.NN_indices_,
    #         compile=self.compile,
    #     )[0]
    #     D = self._a * D**self._b
    #     D = 1 / (2 + D)  # sigmoid trick to avoid numerical instability
    #     return cross_entropy_loss(self.affinity_in_, D)

    def on_affinity_computation_end(self):
        super().on_affinity_computation_end()
        # mask small affinities
        A_max = self.affinity_in_.max()
        threshold = A_max / self.max_iter

        self.epochs_per_sample = torch.where(
            self.affinity_in_ > threshold,
            A_max / self.affinity_in_,
            torch.full_like(self.affinity_in_, fill_value=1e9),
        )
        self.epoch_of_next_sample = self.epochs_per_sample.clone()

    def _compute_attractive_gradients(self):
        D = symmetric_pairwise_distances_indices(
            self.embedding_,
            metric=self.metric_out,
            indices=self.NN_indices_,
            compile=self.compile,
        )[0]
        D = 2 * self._a * self._b * D ** (self._b - 1) / (1 + self._a * D**self._b)

        # consider positive edges with frequency inversely proportional to the affinity value
        self.mask_affinity_in_ = self.epoch_of_next_sample <= (self.n_iter_ + 1)
        self.epoch_of_next_sample = torch.where(
            self.mask_affinity_in_,
            self.epoch_of_next_sample + self.epochs_per_sample,
            self.epoch_of_next_sample,
        )
        D = D * self.mask_affinity_in_  # (n, n_negatives)

        diff = (
            self.embedding_.unsqueeze(1) - self.embedding_[self.NN_indices_]
        )  # (n, n_negatives, d)
        D = torch.clamp(D.unsqueeze(-1) * diff, -4, 4)  # clamp as in umap repo
        return D.sum(dim=1)  # (n, d)

    def _compute_repulsive_gradients(self):
        D = symmetric_pairwise_distances_indices(
            self.embedding_,
            metric=self.metric_out,
            indices=self.neg_indices_,
            compile=self.compile,
        )[0]

        # for each positive edge, we sample 'negative_sample_rate' negative edges
        neg_counts = (
            self.mask_affinity_in_.sum(dim=1) * self.negative_sample_rate
        )  # (n,)
        col_idx = torch.arange(self.n_negatives, device=self.embedding_.device)[
            None, :
        ]  # (1, n_negatives)
        keep = col_idx < neg_counts[:, None]  # (n, n_negatives)
        D = (
            -2 * self._b / ((self._eps + D) * (1 + self._a * D**self._b))
        )  # (n, n_negatives)
        D = D * keep  # (n, n_negatives)

        diff = (
            self.embedding_.unsqueeze(1) - self.embedding_[self.neg_indices_]
        )  # (n, n_negatives, d)
        D = torch.clamp(D.unsqueeze(-1) * diff, -4, 4)  # clamp as in umap repo
        return D.sum(dim=1)  # (n, d)

    def _init_embedding(self, X):
        """
        Initialize the low-dimensional embedding exactly as UMAP does:
         - User-supplied array: rescale so max abs coordinate = 10
         - "random"/"normal": uniform in [-10,10]
         - "pca": top-d PCA, then scale to [-10,10] and add σ=1e-4 jitter
         - "umap_spectral": use original UMAP package with spectral initialization
        """
        n, d = X.shape[0], self.n_components
        device = X.device if self.device == "auto" else self.device
        dtype = torch.float32

        # 1) User-supplied initial embedding
        if isinstance(self.init, (torch.Tensor, np.ndarray)):
            emb = self.init
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb, device=device, dtype=dtype)
            # Rescale so max absolute coordinate is 10
            max_abs = emb.abs().max()
            emb = emb * (10.0 / (max_abs + 1e-12))
            self.embedding_ = emb.requires_grad_()
            return self.embedding_

        # 2) Random uniform in [-10,10]
        if self.init in ("random", "normal"):
            emb = torch.empty((n, d), device=device, dtype=dtype)
            emb.uniform_(-10.0, 10.0)
            self.embedding_ = emb.requires_grad_()
            return self.embedding_

        # 3) PCA + scaling + jitter
        if self.init == "pca":
            from torchdr.spectral_embedding.pca import PCA

            emb = PCA(n_components=d, device=device).fit_transform(X)
            # Rescale so max abs coordinate is 10
            max_abs = emb.abs().max()
            emb = emb * (10.0 / (max_abs + 1e-12))
            # Add tiny Gaussian noise (σ=1e-4)
            emb = emb + torch.randn_like(emb, device=device, dtype=emb.dtype) * 1e-4
            self.embedding_ = emb.requires_grad_()
            return self.embedding_

        # 4) Original UMAP spectral initialization
        if self.init == "umap_spectral":
            try:
                import umap

                # Convert tensor to numpy if needed
                X_np = X.cpu().numpy() if hasattr(X, "cpu") else X

                # Create UMAP instance for initialization only
                umap_init = umap.UMAP(
                    n_neighbors=int(self.n_neighbors),
                    n_components=d,
                    init="spectral",
                    n_epochs=0,  # Don't optimize, just get initialization
                    verbose=False,
                    random_state=self.random_state,
                )

                # Fit to get the spectral initialization
                umap_init.fit(X_np)

                # Get the spectral initialization embedding
                # In UMAP, the spectral init is stored before optimization
                from umap.spectral import spectral_layout

                # Get knn graph like UMAP does
                from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors

                # Get nearest neighbors
                knn_indices, knn_dists, _ = nearest_neighbors(
                    X_np,
                    int(self.n_neighbors),
                    umap_init.metric,
                    {},
                    umap_init.angular_rp_forest,
                    np.random.RandomState(self.random_state),
                    verbose=False,
                )

                # Get fuzzy simplicial set
                graph, _, _, _ = fuzzy_simplicial_set(
                    X_np,
                    int(self.n_neighbors),
                    np.random.RandomState(self.random_state),
                    umap_init.metric,
                    {},
                    knn_indices,
                    knn_dists,
                    umap_init.angular_rp_forest,
                    umap_init.set_op_mix_ratio,
                    umap_init.local_connectivity,
                    verbose=False,
                    return_dists=True,
                )

                # Get spectral initialization
                emb_np = spectral_layout(
                    X_np, graph, d, np.random.RandomState(self.random_state)
                )

                # Convert to tensor and scale like original UMAP
                emb = torch.as_tensor(emb_np, device=device, dtype=dtype)

                # Apply UMAP's scaling: scale to [-10, 10] with noise
                max_abs = emb.abs().max()
                emb = emb * (10.0 / (max_abs + 1e-12))
                emb = emb + torch.randn_like(emb, device=device, dtype=emb.dtype) * 1e-4

                self.embedding_ = emb.requires_grad_()
                return self.embedding_

            except ImportError:
                print(
                    "Warning: umap-learn package not found. Falling back to PCA initialization."
                )
                # Fall back to PCA if umap package not available
                from torchdr.spectral_embedding.pca import PCA

                emb = PCA(n_components=d, device=device).fit_transform(X)
                max_abs = emb.abs().max()
                emb = emb * (10.0 / (max_abs + 1e-12))
                emb = emb + torch.randn_like(emb, device=device, dtype=emb.dtype) * 1e-4
                self.embedding_ = emb.requires_grad_()
                return self.embedding_

        raise ValueError(f"Unsupported init '{self.init}' in _init_embedding")
