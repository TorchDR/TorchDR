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

    def on_affinity_computation_end(self):
        super().on_affinity_computation_end()

        # Remove small affinity edges
        A_max = self.affinity_in_.max()
        threshold = A_max / self.max_iter

        if self.verbose:
            edges_removed = (self.affinity_in_ <= threshold).sum()
            total_edges = self.affinity_in_.numel()
            percentage_kept = (
                (total_edges - edges_removed).float() / total_edges * 100
            ).item()
            self.logger.info(f"Keeping {percentage_kept:.1f}% of affinity edges.")

        self.epochs_per_sample = torch.where(
            self.affinity_in_ > threshold,
            A_max / (self.affinity_in_ + 1e-3),
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
        D = torch.where(
            D > 0,
            2 * self._a * self._b * D ** (self._b - 1) / (1 + self._a * D**self._b),
            torch.zeros_like(D),
        )  # compute D**(b-1) for D > 0 to prevent infinities when b < 1

        # UMAP keeps a per-edge counter (epoch_of_next_sample) so that stronger edges
        # (higher affinity → smaller epochs_per_sample) get updated more often.
        self.mask_affinity_in_ = self.epoch_of_next_sample <= (self.n_iter_ + 1)
        self.epoch_of_next_sample = torch.where(
            self.mask_affinity_in_,
            self.epoch_of_next_sample + self.epochs_per_sample,
            self.epoch_of_next_sample,
        )
        D = D * self.mask_affinity_in_

        embedding_diff = (
            self.embedding_.unsqueeze(1) - self.embedding_[self.NN_indices_]
        )  # (n, n_negatives, d)
        D = D.unsqueeze(-1) * embedding_diff
        D = torch.clamp(D, -4, 4)  # clamp as in umap repo
        return D.sum(dim=1)

    def _compute_repulsive_gradients(self):
        D = symmetric_pairwise_distances_indices(
            self.embedding_,
            metric=self.metric_out,
            indices=self.neg_indices_,
            compile=self.compile,
        )[0]
        D = -2 * self._b / ((self._eps + D) * (1 + self._a * D**self._b))

        # Filter to keep 'negative_sample_rate' negative edges per positive edge.
        neg_counts = self.mask_affinity_in_.sum(dim=1) * self.negative_sample_rate
        col_idx = torch.arange(self.n_negatives, device=self.embedding_.device)[None, :]
        keep = col_idx < neg_counts[:, None]
        D = D * keep

        embedding_diff = (
            self.embedding_.unsqueeze(1) - self.embedding_[self.neg_indices_]
        )  # (n, n_negatives, d)
        D = D.unsqueeze(-1) * embedding_diff
        D = torch.clamp(D, -4, 4)  # clamp as in umap repo
        return D.sum(dim=1)
