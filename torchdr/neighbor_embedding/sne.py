# -*- coding: utf-8 -*-
"""
Stochastic Neighbor embedding (SNE) algorithm
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.affinity_matcher import AffinityMatcher
from torchdr.affinity import (
    EntropicAffinity,
    GibbsAffinity,
)
from torchdr.losses import cross_entropy_loss


class SNE(AffinityMatcher):
    """
    Implementation of the Stochastic Neighbor Embedding (SNE) algorithm
    introduced in [1]_.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    n_components : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z).
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use.
    optimizer_kwargs : dict, optional
        Arguments for the optimizer.
    lr : float, optional
        Learning rate for the algorithm.
    scheduler : {'constant', 'linear'}, optional
        Learning rate scheduler.
    init : {'random', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z.
    metric : {'euclidean', 'manhattan'}, optional
        Metric to use for the affinity computation.
    tol : float, optional
        Precision threshold at which the algorithm stops.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm.
    tol_ea : _type_, optional
        Precision threshold for the entropic affinity root search.
    max_iter_ea : int, optional
        Number of maximum iterations for the entropic affinity root search.
    verbose : bool, optional
        Verbosity, by default True.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary.
    keops : bool, optional
        Whether to use KeOps.

    References
    ----------
    .. [1]  Geoffrey Hinton, Sam Roweis (2002).
            Stochastic Neighbor Embedding.
            Advances in neural information processing systems 15 (NeurIPS).

    .. [2]  Laurens van der Maaten, Geoffrey Hinton (2008).
            Visualizing Data using t-SNE.
            The Journal of Machine Learning Research 9.11 (JMLR).
    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        n_components: int = 2,
        lr: float = 1e0,
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        scheduler: str = "constant",
        init: str = "pca",
        init_scaling: float = 1e-4,
        tol: float = 1e-4,
        max_iter: int = 1000,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "euclidean",
        metric_out: str = "euclidean",
        verbose: bool = True,
        tolog: bool = False,
        device: str = None,
        keops: bool = True,
    ):
        self.metric_in = metric_in
        self.metric_out = metric_out
        self.perplexity = perplexity
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity

        affinity_in = EntropicAffinity(
            perplexity=perplexity,
            metric=metric_in,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        affinity_out = GibbsAffinity(
            metric=metric_out,
            normalization_dim=1,
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

    def _loss(self):
        log_Q = self.affinity_out.fit_transform(self.embedding_, log=True)
        loss = cross_entropy_loss(self.PX_, log_Q, log_Q=True)
        return loss
