# -*- coding: utf-8 -*-
"""SNEkhorn algorithm (inverse OT DR)."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.neighbor_embedding.base import NeighborEmbedding
from torchdr.affinity import (
    SymmetricEntropicAffinity,
    EntropicAffinity,
    SinkhornAffinity,
)
from torchdr.utils import logsumexp_red, cross_entropy_loss


class TSNEkhorn(NeighborEmbedding):
    r"""Implementation of the TSNEkhorn algorithm introduced in [3]_.

    It involves selecting a :class:`~torchdr.SymmetricEntropicAffinity` as input
    affinity :math:`\mathbf{P}` and a :class:`~torchdr.SinkhornAffinity` as output
    affinity :math:`\mathbf{Q}`.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \sum_{ij} Q_{ij} \:.

    The above loss is the gap objective for inverse symmetric optimal transport
    described in this
    `blog <https://huguesva.github.io/blog/2024/inverseOT_mongegap/>`_.

    .. note::
        The :class:`~torchdr.SymmetricEntropicAffinity` requires a careful choice of
        learning rate (parameter :attr:`lr_affinity_in`). If it is too unstable, one
        can resort to using :class:`~torchdr.EntropicAffinity` instead by setting
        :attr:`symmetric_affinity` to ``False``.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    n_components : int, optional
        Dimension of the embedding space.
    lr : float, optional
        Learning rate for the algorithm, by default 1e0.
    optimizer : {'SGD', 'Adam', 'NAdam'}, optional
        Which pytorch optimizer to use, by default 'Adam'.
    optimizer_kwargs : dict, optional
        Arguments for the optimizer, by default None.
    scheduler : {'constant', 'linear'}, optional
        Learning rate scheduler.
    scheduler_kwargs : dict, optional
        Arguments for the scheduler, by default None.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-4.
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
        By default 10.0 for early exaggeration.
    coeff_repulsion : float, optional
        Coefficient for the repulsion term, by default 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default 250.
    lr_affinity_in : float, optional
        Learning rate used to update dual variables for the symmetric entropic
        affinity computation.
    eps_square_affinity_in : bool, optional
        When computing the symmetric entropic affinity, whether to optimize
        on the square of the dual variables. May be more stable in practice.
    tol_affinity_in : _type_, optional
        Precision threshold for the symmetric entropic affinity computation.
    max_iter_affinity_in : int, optional
        Number of maximum iterations for the symmetric entropic affinity computation.
    metric_in : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    metric_out : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the output affinity, by default 'sqeuclidean'.
    unrolling : bool, optional
        Whether to use unrolling for solving inverse OT. If False, uses
        the gap objective. Default is False.
    symmetric_affinity : bool, optional
        Whether to use symmetric entropic affinity. If False, uses
        entropic affinity. Default is True.

    References
    ----------

    .. [3] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities,
        Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty.
        Advances in neural information processing systems 36 (NeurIPS).

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
        max_iter: int = 2000,
        tolog: bool = False,
        device: str = None,
        keops: bool = False,
        verbose: bool = False,
        random_state: float = 0,
        early_exaggeration: float = 10.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = 250,
        lr_affinity_in: float = 1e-1,
        eps_square_affinity_in: bool = True,
        tol_affinity_in: float = 1e-3,
        max_iter_affinity_in: int = 100,
        metric_in: str = "sqeuclidean",
        metric_out: str = "sqeuclidean",
        unrolling: bool = False,
        symmetric_affinity: bool = True,
    ):
        self.metric_in = metric_in
        self.metric_out = metric_out
        self.perplexity = perplexity
        self.lr_affinity_in = lr_affinity_in
        self.eps_square_affinity_in = eps_square_affinity_in
        self.max_iter_affinity_in = max_iter_affinity_in
        self.tol_affinity_in = tol_affinity_in
        self.unrolling = unrolling
        self.symmetric_affinity = symmetric_affinity

        if self.symmetric_affinity:
            affinity_in = SymmetricEntropicAffinity(
                perplexity=perplexity,
                lr=lr_affinity_in,
                eps_square=eps_square_affinity_in,
                metric=metric_in,
                tol=tol_affinity_in,
                max_iter=max_iter_affinity_in,
                device=device,
                keops=keops,
                verbose=verbose,
                zero_diag=False,
            )
        else:
            affinity_in = EntropicAffinity(
                perplexity=perplexity,
                metric=metric_in,
                tol=tol_affinity_in,
                max_iter=max_iter_affinity_in,
                device=device,
                keops=keops,
                verbose=verbose,
            )
        affinity_out = SinkhornAffinity(
            metric=metric_out,
            device=device,
            keops=keops,
            verbose=False,
            base_kernel="student",
            with_grad=unrolling,
            max_iter=5,
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

    def _loss(self):
        if not hasattr(self, "dual_sinkhorn_"):
            self.dual_sinkhorn_ = None

        log_Q = self.affinity_out(
            self.embedding_, log=True, init_dual=self.dual_sinkhorn_
        )
        self.dual_sinkhorn_ = self.affinity_out.dual_.detach()
        P = self.PX_

        attractive_term = cross_entropy_loss(P, log_Q, log=True)
        if self.unrolling:
            repulsive_term = 0
        else:
            repulsive_term = logsumexp_red(log_Q, dim=(0, 1)).exp()

        loss = (
            self.early_exaggeration_ * attractive_term
            + self.coeff_repulsion * repulsive_term
        )
        return loss
