# -*- coding: utf-8 -*-
"""
Stochastic Neighbor embedding algorithms (SNE, tSNE)
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.affinity_matcher import AffinityMatcher
from torchdr.affinity import (
    EntropicAffinity,
    GibbsAffinity,
)
from torchdr.utils import cross_entropy_loss


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
        Whether to use KeOps for the entropic affinity computation.

    Attributes
    ----------
    log_ : dictionary
        Contains the log of affinity_embedding, affinity_data and the loss at each iteration (if tolog is True).
    n_iter_: int
        Number of iterations run.
    embedding_ : torch.Tensor of shape (n_samples, n_components)
        Stores the embedding coordinates.
    PX_ :  torch.Tensor of shape (n_samples, n_samples)
        Fitted entropic affinity matrix in the input space.

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
        perplexity=10,
        n_components=2,
        optimizer="Adam",
        optimizer_kwargs={},
        lr=1.0,
        scheduler="constant",
        init="pca",
        init_scaling=1e-4,
        metric="euclidean",
        tol=1e-4,
        max_iter=1000,
        tol_ea=1e-3,
        max_iter_ea=100,
        verbose=True,
        tolog=False,
        device=None,
        keops=True,
    ):
        affinity_input = EntropicAffinity(
            perplexity=perplexity,
            metric=metric,
            tol=tol_ea,
            max_iter=max_iter_ea,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        affinity_embedding = GibbsAffinity(
            metric=metric,
            normalization_dim=1,
            device=device,
            keops=keops,
            verbose=False,
        )

        super().__init__(
            affinity_data=affinity_input,
            affinity_embedding=affinity_embedding,
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
        log_Q = self.affinity_embedding.fit_transform(self.embedding_, log=True)
        loss = cross_entropy_loss(self.PX_, log_Q, log_Q=True)
        return loss
