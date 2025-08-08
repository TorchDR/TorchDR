# -*- coding: utf-8 -*-
"""Hyperbolic Stochastic Neighbor Embedding (CO-SNE) algorithm."""

# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Union, Type, Any
import torch

from torchdr.neighbor_embedding.base import SparseNeighborEmbedding
from torchdr.affinity import (
    EntropicAffinity,
    CauchyAffinity,
)
from torchdr.utils import logsumexp_red, RiemannianAdam


class COSNE(SparseNeighborEmbedding):
    """Implementation of the CO-Stochastic Neighbor Embedding (CO-SNE) introduced in :cite:`guo2022co`.

    This algorithm is a variant of SNE that uses a hyperbolic space for the embedding.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    lambda1 : float
        Coefficient for the loss enforcing equal norms between input samples
        and embedded samples.
    gamma : float
        Gamma parameter of the Cauchy distribution used for affinity, by default 2.
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
    init : {'hyperbolic'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'hyperbolic'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 0.5.
    tol : float, optional
        Precision threshold at which the algorithm stops, by default 1e-4.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 2000.
    device : str, optional
        Device to use, by default "auto".
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity, by default False.
    random_state : float, optional
        Random seed for reproducibility, by default None.
    early_exaggeration_coeff : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        By default 12.0 for early exaggeration.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default 250.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric_in : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    sparsity : bool, optional
        Whether to use sparsity mode for the input affinity. Default is True.
    check_interval : int, optional
        Number of iterations between checks for convergence, by default 50.
    precision : {"32-true", "16-mixed", "bf16-mixed", 32, 16}, optional
        Precision mode for affinity and gradient computations. Default is "32-true".
        - "32-true" or 32: Full precision (float32)
        - "16-mixed" or 16: Mixed precision with float16
        - "bf16-mixed": Mixed precision with bfloat16
    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        lambda1: float = 1,
        gamma: float = 2,
        n_components: int = 2,
        lr: Union[float, str] = "auto",
        optimizer_kwargs: Optional[Union[Dict, str]] = None,
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        init: str = "hyperbolic",
        init_scaling: float = 0.5,
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 12.0,
        early_exaggeration_iter: Optional[int] = 250,
        max_iter_affinity: int = 100,
        metric_in: str = "sqeuclidean",
        sparsity: bool = True,
        check_interval: int = 50,
        compile: bool = False,
        precision: Union[str, int] = "32-true",
        **kwargs,
    ):
        self.metric_in = metric_in
        self.metric_out = "sqhyperbolic"
        self.perplexity = perplexity
        self.lambda1 = lambda1
        self.gamma = gamma
        self.max_iter_affinity = max_iter_affinity
        self.sparsity = sparsity

        affinity_in = EntropicAffinity(
            perplexity=perplexity,
            metric=metric_in,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=sparsity,
            precision=precision,
        )
        affinity_out = CauchyAffinity(
            metric=self.metric_out,
            gamma=gamma,
            device=device,
            backend=backend,
            verbose=verbose,
            precision=precision,
        )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            n_components=n_components,
            optimizer=RiemannianAdam,
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
            compile=compile,
            precision=precision,
            **kwargs,
        )

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        # We compute once and for all the norms of X data samples
        self.X_norm = (X**2).sum(-1)
        return super()._fit_transform(X)

    def _compute_repulsive_loss(self):
        log_Q = self.affinity_out(self.embedding_, log=True)
        rep_loss = logsumexp_red(log_Q, dim=(0, 1))  # torch.tensor([0])
        Y_norm = (self.embedding_**2).sum(-1)
        Y_norm = torch.arccosh(1 + 2 * (Y_norm / (1 - Y_norm)) + 1e-8) ** 2
        distance_term = ((self.X_norm - Y_norm) ** 2).mean()
        return rep_loss + self.lambda1 * distance_term
