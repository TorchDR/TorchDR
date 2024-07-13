# -*- coding: utf-8 -*-
"""
UMAP algorithm
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from torchdr.affinity import (
    UMAPAffinityIn,
    UMAPAffinityOut,
)
from torchdr.utils import sum_all_axis_except_batch, cross_entropy_loss


class UMAP(SampledNeighborEmbedding):
    """
    Implementation of the UMAP algorithm introduced in [8]_ and further studied
    in [12]_.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors.
    n_components : int, optional
        Dimension of the embedding space.
    min_dist : float, optional
        Minimum distance between points in the embedding space.
    spread : float, optional
        Initial spread of the embedding space.
    a : float, optional
        Parameter for the Student t-distribution.
    b : float, optional
        Parameter for the Student t-distribution.
    lr : float, optional
        Learning rate for the algorithm, by default 1.0.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use, by default 'Adam'.
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
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm.
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
        Coefficient for the attraction term, by default 1.0.
    coeff_repulsion : float, optional
        Coefficient for the repulsion term, by default 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default 250.
    tol_affinity : float, optional
        Precision threshold for the input affinity computation.
    max_iter_affinity : int, optional
        Number of maximum iterations for the input affinity computation.
    metric_in : {'euclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'euclidean'.
    metric_out : {'euclidean', 'manhattan'}, optional
        Metric to use for the output affinity, by default 'euclidean'.
    n_negatives : int, optional
        Number of negative samples for the noise-contrastive loss, by default 5.

    References
    ----------
    .. [8] Leland McInnes, John Healy, James Melville (2018).
        UMAP: Uniform manifold approximation and projection for dimension reduction.
        arXiv preprint arXiv:1802.03426.

    .. [12] Sebastian Damrich, Fred Hamprecht (2021).
        On UMAP's True Loss Function.
        Advances in Neural Information Processing Systems 34 (NeurIPS).

    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 30,
        n_components: int = 2,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float = None,
        b: float = None,
        lr: float = 1.0,
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        tol: float = 1e-7,
        max_iter: int = 1000,
        tolog: bool = False,
        device: str = None,
        keops: bool = False,
        verbose: bool = True,
        random_state: float = 0,
        coeff_attraction: float = 1.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = 0,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "sqeuclidean",
        metric_out: str = "sqeuclidean",
        n_negatives: int = 5,
    ):

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.a = a
        self.b = b
        self.metric_in = metric_in
        self.metric_out = metric_out
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity

        affinity_in = UMAPAffinityIn(
            n_neighbors=n_neighbors,
            metric=metric_in,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        affinity_out = UMAPAffinityOut(
            min_dist=min_dist,
            spread=spread,
            a=a,
            b=b,
            metric=metric_out,
            device=device,
            keops=keops,
            verbose=False,
        )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            tol=tol,
            max_iter=max_iter,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
            coeff_attraction=coeff_attraction,
            coeff_repulsion=coeff_repulsion,
            early_exaggeration_iter=early_exaggeration_iter,
            n_negatives=n_negatives,
        )

    @sum_all_axis_except_batch
    def _repulsive_loss(self):
        indices = self._sample_negatives()
        Q = self.affinity_out(self.embedding_, indices=indices)
        Q = Q / (Q + 1)  # stabilization trick, PR #856 from UMAP repo
        return -(1 - Q).log()

    def _attractive_loss(self):
        Q = self.affinity_out(self.embedding_, indices=self.indices_)
        Q = Q / (Q + 1)  # stabilization trick, PR #856 from UMAP repo
        return cross_entropy_loss(self.PX_, Q)
