# -*- coding: utf-8 -*-
"""
Hyperbolic Stochastic Neighbor embedding (CO-SNE) algorithm
"""

# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Union, Type
import torch

from torchdr.neighbor_embedding.base import SparseNeighborEmbedding
from torchdr.affinity import (
    EntropicAffinity,
    CauchyAffinity,
)
from torchdr.utils import (
    logsumexp_red,
    geoopt,
    is_geoopt_available
)


class COSNE(SparseNeighborEmbedding):
    """
    Implementation of the CO-Stochastic Neighbor Embedding (CO-SNE) algorithm
    for embedding in hyperbolic space, introduced in [23]_.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    lambda1 : float
        Coefficient for the loss enforcing equal norms between input samples
        and embedded samples
    gamma : float
        gamma parameter of the Cauchy distribution used for affinity, by default 2
    n_components : int, optional
        Dimension of the embedding space.
    lr : float, optional
        Learning rate for the algorithm, by default 1.0.
    optimizer_kwargs : dict, optional
        Arguments for the optimizer, by default None.
    scheduler : {'constant', 'linear'}, optional
        Learning rate scheduler.
    scheduler_kwargs : dict, optional
        Arguments for the scheduler, by default None.
    init : {'random', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-4.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary, by default False.
    device : str, optional
        Device to use, by default "auto".
    keops : bool, optional
        Whether to use KeOps, by default False.
    verbose : bool, optional
        Verbosity, by default True.
    random_state : float, optional
        Random seed for reproducibility, by default 0.
    coeff_attraction : float, optional
        Coefficient for the attraction term, by default 10.0 for early exaggeration.
    coeff_repulsion : float, optional
        Coefficient for the repulsion term, by default 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default 250.
    tol_affinity : _type_, optional
        Precision threshold for the entropic affinity root search.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric_in : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.

    References
    ----------

    .. [23]  Guo, Y., Guo, H. & Yu, S. (2022).
            `CO-SNE: Dimensionality Reduction and Visualization for Hyperbolic Data`_.
            International Conference on Computer Vision and Pattern Recognition (CVPR).


    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        lambda1: float = 1,
        gamma: float = 2,
        n_components: int = 2,
        lr: Union[float, str] = "auto",
        optimizer_kwargs: Union[Dict, str] = {},
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        init: str = "hyperbolic",
        init_scaling: float = 1,
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 12.0,
        early_exaggeration_iter: Optional[int] = 250,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "sqeuclidean",
        sparsity: bool = True,
    ):
        if is_geoopt_available():
            self.metric_in = metric_in
            self.metric_out = "sqhyperbolic"
            self.perplexity = perplexity
            self.lambda1 = lambda1
            self.gamma = gamma
            self.max_iter_affinity = max_iter_affinity
            self.tol_affinity = tol_affinity
            self.sparsity = sparsity

            affinity_in = EntropicAffinity(
                perplexity=perplexity,
                metric=metric_in,
                tol=tol_affinity,
                max_iter=max_iter_affinity,
                device=device,
                backend=backend,
                verbose=verbose,
                sparsity=sparsity,
            )
            affinity_out = CauchyAffinity(
                metric=self.metric_out,
                gamma=self.gamma,
                device=device,
                backend=backend,
                verbose=False,
            )

            super().__init__(
                affinity_in=affinity_in,
                affinity_out=affinity_out,
                n_components=n_components,
                optimizer=geoopt.optim.RiemannianAdam,
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
            )

    def _fit(self, X: torch.Tensor):
        # We compute once for all the norms of X data samples
        self.X_norm = (X**2).sum(-1)
        super()._fit(X)

    def _repulsive_loss(self):
        ball = geoopt.PoincareBall()
        ball.assert_check_point_on_manifold(self.embedding_)
        log_Q = self.affinity_out(self.embedding_, log=True)
        rep_loss = logsumexp_red(log_Q, dim=(0, 1))
        Y_norm = (self.embedding_**2).sum(-1)
        distance_term = ((self.X_norm - Y_norm)**2).sum()
        return rep_loss + self.lambda1 * distance_term
