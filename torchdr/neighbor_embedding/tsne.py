# -*- coding: utf-8 -*-
"""t-distributed Stochastic Neighbor embedding (TSNE) algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.neighbor_embedding.base import SparseNeighborEmbedding
from torchdr.affinity import (
    EntropicAffinity,
    StudentAffinity,
)
from torchdr.utils import logsumexp_red


class TSNE(SparseNeighborEmbedding):
    r"""Implementation of t-Stochastic Neighbor Embedding (t-SNE) introduced in [2]_.

    It involves selecting a :class:`~torchdr.EntropicAffinity` as input
    affinity :math:`\mathbf{P}` and a :class:`~torchdr.StudentAffinity` as output
    affinity :math:`\mathbf{Q}`.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \log \Big( \sum_{ij} Q_{ij} \Big) \:.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    n_components : int, optional
        Dimension of the embedding space.
    lr : float or 'auto', optional
        Learning rate for the algorithm, by default 'auto'.
    optimizer : {'SGD', 'Adam', 'NAdam', 'auto}, optional
        Which pytorch optimizer to use, by default 'auto'.
    optimizer_kwargs : dict or 'auto', optional
        Arguments for the optimizer, by default 'auto'.
    scheduler : {'constant', 'linear'}, optional
        Learning rate scheduler.
    scheduler_kwargs : dict, optional
        Arguments for the scheduler, by default None.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 2000.
    tolog : bool, optional
        Whether to store intermediate results in a dictionary, by default False.
    device : str, optional
        Device to use, by default "auto".
    keops : bool, optional
        Whether to use KeOps, by default False.
    verbose : bool, optional
        Verbosity, by default False.
    random_state : float, optional
        Random seed for reproducibility, by default 0.
    early_exaggeration : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        By default 12.0 for early exaggeration.
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
    metric_out : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the output affinity, by default 'sqeuclidean'.

    References
    ----------

    .. [2]  Laurens van der Maaten, Geoffrey Hinton (2008).
            Visualizing Data using t-SNE.
            The Journal of Machine Learning Research 9.11 (JMLR).

    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        n_components: int = 2,
        lr: float | str = "auto",
        optimizer: str = "auto",
        optimizer_kwargs: dict | str = "auto",
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        tol: float = 1e-7,
        max_iter: int = 2000,
        tolog: bool = False,
        device: str = None,
        keops: bool = False,
        verbose: bool = False,
        random_state: float = 0,
        early_exaggeration: float = 12.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = 250,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "sqeuclidean",
        metric_out: str = "sqeuclidean",
        **kwargs,
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
        affinity_out = StudentAffinity(
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
            early_exaggeration=early_exaggeration,
            coeff_repulsion=coeff_repulsion,
            early_exaggeration_iter=early_exaggeration_iter,
        )

    def _repulsive_loss(self):
        log_Q = self.affinity_out(self.embedding_, log=True)
        return logsumexp_red(log_Q, dim=(0, 1))
