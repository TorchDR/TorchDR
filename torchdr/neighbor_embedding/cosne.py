# -*- coding: utf-8 -*-
"""
Hyperbolic Stochastic Neighbor embedding (CO-SNE) algorithm
"""

# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

from torchdr.affinity_matcher import AffinityMatcher
from torchdr.affinity import (
    L2SymmetricEntropicAffinity,
    StudentAffinity,
)
from torchdr.losses import cross_entropy_loss
from torchdr.utils import logsumexp_red


class COSNE(AffinityMatcher):
    """
    Implementation of the CO-Stochastic Neighbor Embedding (CO-SNE) algorithm
    for embedding in hyperbolic space, introduced in [23]_.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    n_components : int, optional
        Dimension of the embedding space.
    lr : float, optional
        Learning rate for the algorithm, by default 1.0.
    optimizer : {'RAdam'}, optional
        Which pytorch/Geoopt optimizer to use, by default 'RAdam'.
    optimizer_kwargs : dict, optional
        Arguments for the optimizer, by default None.
    scheduler : {'constant', 'linear'}, optional
        Learning rate scheduler.
    init : {'random', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    metric_in : {'euclidean', 'manhattan'}, optional
        Metric to use for the affinity computation, by default 'euclidean'.
    metric_out : {'euclidean', 'manhattan'}, optional
        Metric to use for the affinity computation, by default 'euclidean'.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-4.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 100.
    tol_affinity : _type_, optional
        Precision threshold for the entropic affinity root search.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary, by default False.
    device : str, optional
        Device to use, by default None.
    keops : bool, optional
        Whether to use KeOps, by default True.
    early_exaggeration : int, optional
        Early exaggeration factor, by default 12.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default 250.
    verbose : bool, optional
        Verbosity, by default True.

    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        n_components: int = 2,
        lr: float = 1.0,
        optimizer: str = "RAdam",
        optimizer_kwargs: dict = None,
        scheduler: str = "constant",
        init: str = "pca",
        init_scaling: float = 1e-4,
        metric_in: str = "euclidean",
        metric_out: str = "euclidean",
        tol: float = 1e-4,
        max_iter: int = 1000,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        tolog: bool = False,
        device: str = None,
        keops: bool = True,
        early_exaggeration: int = 12,
        early_exaggeration_iter: int = 250,
        verbose: bool = True,
    ):

        self.metric_in = metric_in
        self.metric_out = metric_out
        self.perplexity = perplexity
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity

        affinity_in = L2SymmetricEntropicAffinity(
            perplexity=perplexity,
            metric=metric_in,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        affinity_out = StudentAffinity(
            metric=metric_out,
            normalization_dim=None,  # we perform normalization when computing the loss
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
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
        )

        if early_exaggeration is None or early_exaggeration_iter is None:
            self.early_exaggeration = 1
            early_exaggeration_iter = None
        else:
            self.early_exaggeration = early_exaggeration
            self.early_exaggeration_iter = early_exaggeration_iter

    def _loss(self):
        """
        Dimensionality reduction objective.
        """
        log_Q = self.affinity_out.fit_transform(self.embedding_, log=True)
        attractive_term = cross_entropy_loss(self.PX_, log_Q, log_Q=True)
        repulsive_term = logsumexp_red(log_Q, dim=(0, 1))
        loss = self.early_exaggeration_ * attractive_term + repulsive_term
        return loss
