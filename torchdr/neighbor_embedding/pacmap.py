"""PACMAP algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from typing import Union, Optional, Dict, Type, Any
from torchdr.affinity import PACMAPAffinity
from torchdr.utils import kmin, sum_red
from torchdr.distance import pairwise_distances, FaissConfig


class PACMAP(SampledNeighborEmbedding):
    r"""PACMAP algorithm introduced in :cite:`wang2021understanding`.

    It uses a :class:`~torchdr.PACMAPAffinity` as input affinity.
    The loss function is defined as:

    .. math::

        w_{\mathrm{NB}} \sum_{i, j \in \mathrm{NB}(i)} \frac{d_{ij}}{10 + d_{ij}} + w_{\mathrm{MN}} \sum_{i,j \in \mathrm{MN}(i)} \frac{d_{ij}}{10^4 + d_{ij}} + w_{\mathrm{FP}} \sum_{i,j \in \mathrm{FP}(i)} \frac{1}{1 + d_{ij}}

    where :math:`\mathrm{NB}(i)`, :math:`\mathrm{MN}(i)` and :math:`\mathrm{FP}(i)` are the nearest neighbors, mid-near neighbors and far neighbors of point :math:`i` respectively,
    and :math:`d_{ij} = 1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2` (more details in :cite:`wang2021understanding`).

    Parameters
    ----------
    n_neighbors : int, optional
        Number of nearest neighbors.
    n_components : int, optional
        Dimension of the embedding space.
    lr : float or 'auto', optional
        Learning rate for the algorithm, by default 1e0.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "Adam".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is None,
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    min_grad_norm : float, optional
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 450.
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
    metric : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    MN_ratio : float, optional
        Ratio of mid-near pairs to nearest neighbor pairs, by default 0.5.
    FP_ratio : float, optional
        Ratio of far pairs to nearest neighbor pairs, by default 2.
    check_interval : int, optional
        Interval for checking convergence, by default 50.
    iter_per_phase : int, optional
        Number of iterations for each phase of the algorithm, by default 100.
    discard_NNs : bool, optional
        Whether to discard the nearest neighbors from the negative sampling.
        Default is True.
    compile : bool, optional
        Whether to compile the algorithm using torch.compile. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 10,
        n_components: int = 2,
        lr: Union[float, str] = 1e0,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        optimizer_kwargs: Optional[Union[Dict, str]] = None,
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        min_grad_norm: float = 1e-7,
        max_iter: int = 450,
        device: Optional[str] = None,
        backend: Union[str, FaissConfig, None] = "faiss",
        verbose: bool = False,
        random_state: Optional[float] = None,
        metric: str = "sqeuclidean",
        MN_ratio: float = 0.5,
        FP_ratio: float = 2,
        check_interval: int = 50,
        iter_per_phase: int = 100,
        discard_NNs: bool = True,
        compile: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric

        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.n_mid_near = int(MN_ratio * n_neighbors)
        self.n_further = int(FP_ratio * n_neighbors)
        self.iter_per_phase = iter_per_phase

        affinity_in = PACMAPAffinity(
            n_neighbors=n_neighbors,
            metric=metric,
            device=device,
            backend=backend,
            verbose=verbose,
        )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=None,
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
            n_negatives=self.n_further,
            discard_NNs=discard_NNs,
            compile=compile,
        )

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None):
        # Keep input data to compute mid-near loss
        target_device = self._get_compute_device(X)
        X_on_device = X.to(target_device) if target_device != X.device else X
        self.register_buffer("X_", X_on_device, persistent=False)
        self._set_weights()
        self_idxs = torch.arange(self.X_.shape[0], device=target_device).unsqueeze(1)
        self.register_buffer("self_idxs", self_idxs, persistent=False)

        mid_near_indices = torch.empty(
            self.X_.shape[0], self.n_mid_near, device=target_device
        )
        self.register_buffer("mid_near_indices", mid_near_indices, persistent=False)
        return super()._fit_transform(X, y)

    def _set_weights(self):
        if self.n_iter_ < self.iter_per_phase:
            self.w_NB = 2
            self.w_MN = (
                1000 * (1 - self.n_iter_ / self.iter_per_phase)
                + 3 * self.n_iter_ / self.iter_per_phase
            )
            self.w_FP = 1
        elif self.n_iter_ < 2 * self.iter_per_phase:
            self.w_NB = 3
            self.w_MN = 3
            self.w_FP = 1
        else:
            self.w_NB = 1
            self.w_MN = 0
            self.w_FP = 1

    def on_training_step_end(self):
        self._set_weights()

    def _compute_attractive_loss(self):
        # Attractive loss with nearest neighbors
        Q_near = 1 + pairwise_distances(
            self.embedding_,
            metric="sqeuclidean",
            backend=self.backend,
            indices=self.NN_indices_,
        )
        Q_near = Q_near / (10 + Q_near)
        near_loss = self.w_NB * sum_red(Q_near, dim=(0, 1))

        if self.w_MN > 0:
            # Attractive loss with mid-near points :
            # we sample 6 mid-near points for each sample
            # and keep the second closest in terms of input space distance
            n_possible_idxs = self.n_samples_in_ - 1

            if n_possible_idxs < 6:
                raise ValueError(
                    "[TorchDR] ERROR : Not enough points to sample 6 mid-near points."
                )

            for i in range(self.n_mid_near):  # to do: broadcast for efficiency
                mid_near_candidates_indices = torch.randint(
                    1,
                    n_possible_idxs,
                    (self.n_samples_in_, 6),
                    device=getattr(self.NN_indices_, "device", "cpu"),
                )
                shifts = torch.searchsorted(
                    self.self_idxs, mid_near_candidates_indices, right=True
                )
                mid_near_candidates_indices.add_(shifts)
                D_mid_near_candidates = pairwise_distances(
                    self.X_,
                    metric=self.metric,
                    backend=self.backend,
                    indices=mid_near_candidates_indices,
                    device=self.device,
                )
                _, idxs = kmin(D_mid_near_candidates, k=2, dim=1)
                self.mid_near_indices[:, i] = idxs[
                    :, 1
                ]  # Retrieve the second closest point

            Q_mid_near = 1 + pairwise_distances(
                self.embedding_,
                metric="sqeuclidean",
                backend=self.backend,
                indices=self.mid_near_indices,
            )
            Q_mid_near = Q_mid_near / (1e4 + Q_mid_near)
            mid_near_loss = self.w_MN * sum_red(Q_mid_near, dim=(0, 1))
        else:
            mid_near_loss = 0

        return near_loss + mid_near_loss

    def _compute_repulsive_loss(self):
        Q_further = 1 + pairwise_distances(
            self.embedding_,
            metric="sqeuclidean",
            backend=self.backend,
            indices=self.neg_indices_,
        )
        Q_further = 1 / (1 + Q_further)
        return self.w_FP * sum_red(Q_further, dim=(0, 1))
