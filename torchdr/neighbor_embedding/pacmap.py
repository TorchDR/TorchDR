"""PACMAP algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from typing import Union, Optional, Dict, Type
from torchdr.affinity import PACMAPAffinity, NegativeCostAffinity
from torchdr.utils import cross_entropy_loss
from torchdr.utils import sum_output


class PACMAP(SampledNeighborEmbedding):
    r"""PACMAP algorithm introduced in :cite:`wang2021understanding`.


    Parameters
    ----------
    n_neighbors : int, optional
        Number of nearest neighbors.
    n_components : int, optional
        Dimension of the embedding space.
    lr : float or 'auto', optional
        Learning rate for the algorithm, by default 'auto'.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto',
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
        Number of maximum iterations for the descent algorithm, by default 3000.
    device : str, optional
        Device to use, by default "auto".
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is "faiss".
    verbose : bool, optional
        Verbosity, by default False.
    random_state : float, optional
        Random seed for reproducibility, by default None.
    early_exaggeration_coeff : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        By default 12.0 for early exaggeration.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default 250.
    tol_affinity : float, optional
        Precision threshold for the entropic affinity root search.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric_in : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    metric_out : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the output affinity, by default 'sqeuclidean'.
    check_interval : int, optional
        Interval for checking convergence, by default 50.
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 10,
        n_components: int = 2,
        lr: Union[float, str] = "auto",
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = "auto",
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        min_grad_norm: float = 1e-7,
        max_iter: int = 3000,
        device: Optional[str] = None,
        backend: Optional[str] = "faiss",
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 12.0,
        early_exaggeration_iter: Optional[int] = 250,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "sqeuclidean",
        metric_out: str = "sqeuclidean",
        MN_ratio: float = 0.5,
        FP_ratio: float = 2,
        check_interval: int = 50,
    ):
        self.n_neighbors = n_neighbors
        self.metric_in = metric_in
        self.metric_out = metric_out
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity

        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.n_mid_near = int(MN_ratio * n_neighbors)

        affinity_in = PACMAPAffinity(
            n_neighbors=n_neighbors,
            metric=metric_in,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        affinity_out = NegativeCostAffinity(
            metric=metric_out,
            device=device,
            verbose=False,
        )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
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
            early_exaggeration_coeff=early_exaggeration_coeff,
            early_exaggeration_iter=early_exaggeration_iter,
            check_interval=check_interval,
            n_negatives=int(FP_ratio * n_neighbors),
        )
        self.mid_near_input_affinity = NegativeCostAffinity(
            metric=metric_in,
            device=device,
            verbose=False,
        )  # used to compute the mid-near points

    def _attractive_loss(self):
        D_tilde_neighbors = 1 - self.affinity_out(  # distance is negative affinity
            self.embedding_, indices=self.NN_indices_
        )
        Q_neighbors = D_tilde_neighbors / (10 + D_tilde_neighbors)
        near_loss = cross_entropy_loss(self.PX_, Q_neighbors, log=False)

        mid_near_loss = cross_entropy_loss(
            self.mid_near_input_affinity(self.embedding_, indices=self.NN_indices_),
            Q_neighbors,
            log=False,
        )
        return self.w_NB * near_loss + self.w_MN * mid_near_loss

    @sum_output
    def _repulsive_loss(self):
        indices = self._sample_negatives()
        D_tilde_further = 1 - self.affinity_out(self.embedding_, indices=indices)
        Q_further = D_tilde_further / (1 + D_tilde_further)
        return self.w_FP * Q_further
