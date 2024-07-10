# -*- coding: utf-8 -*-
"""
SNEkhorn algorithm (inverse OT DR)
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.neighbor_embedding.base import NeighborEmbedding
from torchdr.affinity import (
    SymmetricEntropicAffinity,
    SinkhornAffinity,
)
from torchdr.utils import logsumexp_red, sum_red


class SNEkhorn(NeighborEmbedding):
    """
    Implementation of the SNEkhorn algorithm introduced in [3]_.

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
        max_iter: int = 1000,
        tolog: bool = False,
        device: str = None,
        keops: bool = False,
        verbose: bool = True,
        random_state: float = 0,
        coeff_attraction: float = 10.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = 250,
        lr_affinity_in: float = 1e0,
        eps_square_affinity_in: bool = True,
        tol_affinity_in: float = 1e-3,
        max_iter_affinity_in: int = 100,
        metric_in: str = "sqeuclidean",
        metric_out: str = "sqeuclidean",
        unrolling: bool = False,
    ):

        self.metric_in = metric_in
        self.metric_out = metric_out
        self.perplexity = perplexity
        self.lr_affinity_in = lr_affinity_in
        self.eps_square_affinity_in = eps_square_affinity_in
        self.max_iter_affinity_in = max_iter_affinity_in
        self.tol_affinity_in = tol_affinity_in
        self.unrolling = unrolling

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
        )
        affinity_out = SinkhornAffinity(
            metric=metric_out,
            device=device,
            keops=keops,
            verbose=False,
            base_kernel="gaussian",
            with_grad=unrolling,
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

    def _repulsive_loss(self, Q, log=True):
        if self.unrolling:
            return 0
        else:
            if log:
                return logsumexp_red(Q, dim=(0, 1)).exp()
            else:
                return sum_red(Q, dim=(0, 1))


class TSNEkhorn(NeighborEmbedding):
    """
    Implementation of the TSNEkhorn algorithm introduced in [3]_.

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
        max_iter: int = 1000,
        tolog: bool = False,
        device: str = None,
        keops: bool = False,
        verbose: bool = True,
        random_state: float = 0,
        coeff_attraction: float = 10.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = 250,
        lr_affinity_in: float = 1e0,
        eps_square_affinity_in: bool = True,
        tol_affinity_in: float = 1e-3,
        max_iter_affinity_in: int = 100,
        metric_in: str = "sqeuclidean",
        metric_out: str = "sqeuclidean",
        unrolling: bool = False,
    ):

        self.metric_in = metric_in
        self.metric_out = metric_out
        self.perplexity = perplexity
        self.lr_affinity_in = lr_affinity_in
        self.eps_square_affinity_in = eps_square_affinity_in
        self.max_iter_affinity_in = max_iter_affinity_in
        self.tol_affinity_in = tol_affinity_in
        self.unrolling = unrolling

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
        )
        affinity_out = SinkhornAffinity(
            metric=metric_out,
            device=device,
            keops=keops,
            verbose=False,
            base_kernel="student",
            with_grad=unrolling,
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

    def _repulsive_loss(self, Q, log=True):
        if self.unrolling:
            return 0
        else:
            if log:
                return logsumexp_red(Q, dim=(0, 1)).exp()
            else:
                return sum_red(Q, dim=(0, 1))
