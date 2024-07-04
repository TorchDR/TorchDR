# -*- coding: utf-8 -*-
"""
Stochastic Neighbor embedding (SNE) algorithm
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.neighbor_embedding.base import NeighborEmbedding
from torchdr.affinity import (
    EntropicAffinity,
    GibbsAffinity,
)
from torchdr.utils import logsumexp_red


class SNE(NeighborEmbedding):
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
    lr : float, optional
        Learning rate for the algorithm.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use.
    optimizer_kwargs : dict, optional
        Arguments for the optimizer.
    scheduler : {'constant', 'linear'}, optional
        Learning rate scheduler.
    scheduler_kwargs : dict, optional
        Arguments for the scheduler.
    init : {'random', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z.
    init_scaling : float, optional
        Scaling factor for the initialization.
    tol : float, optional
        Precision threshold at which the algorithm stops.
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
        Coefficient for the attraction term, by default 10.0 for early exaggeration.
    coeff_repulsion : float, optional
        Coefficient for the repulsion term, by default 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default 250.
    tol_affinity : float, optional
        Precision threshold for the entropic affinity root search.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric_in : {'euclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'euclidean'.
    metric_out : {'euclidean', 'manhattan'}, optional
        Metric to use for the output computation, by default 'euclidean'.

    References
    ----------
    .. [1]  Geoffrey Hinton, Sam Roweis (2002).
            Stochastic Neighbor Embedding.
            Advances in neural information processing systems 15 (NeurIPS).

    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        n_components: int = 2,
        lr: float = 1.0,
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        tol: float = 1e-4,
        max_iter: int = 1000,
        tolog: bool = False,
        device: str = None,
        keops: bool = False,
        verbose: bool = True,
        random_state: float = 0,
        coeff_attraction: float = 10.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = 250,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "euclidean",
        metric_out: str = "euclidean",
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
            normalization_dim=None,  # normalization is the repulsive loss
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
        )

    def _repulsive_loss(self, log_Q):
        return logsumexp_red(log_Q, dim=1).sum() / self.n_samples_in_
