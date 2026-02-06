# -*- coding: utf-8 -*-
"""Hyperbolic Stochastic Neighbor Embedding (CO-SNE) algorithm."""

# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Union, Type, Any
import torch
import torch.nn as nn

from torchdr.neighbor_embedding.base import SparseNeighborEmbedding
from torchdr.affinity import EntropicAffinity
from torchdr.distance import FaissConfig, pairwise_distances, pairwise_distances_indexed
from torchdr.utils import (
    logsumexp_red,
    RiemannianAdam,
    cross_entropy_loss,
    PoincareBallManifold,
)


class _ExpMap0(nn.Module):
    """Project Euclidean vectors onto the Poincaré ball via expmap0.

    Clamps the output norm to stay safely inside the ball, avoiding
    numerical issues with hyperbolic distances in float32.
    """

    def __init__(self, manifold, c=1, max_norm=1.0 - 1e-5):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.max_norm = max_norm

    def forward(self, x):
        # Use float64 for numerical stability with hyperbolic distances,
        # matching non-encoder COSNE which initializes embedding in float64.
        z = self.manifold.expmap0(x.double(), c=self.c)
        # Clamp norm to stay safely inside the Poincaré ball
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        z = torch.where(norm > self.max_norm, z * self.max_norm / norm, z)
        return z


class COSNE(SparseNeighborEmbedding):
    r"""Implementation of the CO-Stochastic Neighbor Embedding (CO-SNE) introduced in :cite:`guo2022co`.

    This algorithm is a variant of SNE that uses a hyperbolic space for the embedding.
    It uses a :class:`~torchdr.EntropicAffinity` as input affinity :math:`\mathbf{P}`
    and a Cauchy kernel in hyperbolic space as output affinity :math:`Q_{ij} = \gamma / (d_H(\mathbf{z}_i, \mathbf{z}_j) + \gamma^2)` where :math:`d_H` is the hyperbolic distance.

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
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
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
    metric : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    sparsity : bool, optional
        Whether to use sparsity mode for the input affinity. Default is True.
    check_interval : int, optional
        Number of iterations between checks for convergence, by default 50.
    encoder : torch.nn.Module, optional
        A neural network that maps input data to the embedding space.
        The encoder output is automatically projected onto the Poincaré ball
        via the exponential map. Default is None.
    batch_size : int, optional
        Mini-batch size for encoder-based training. The repulsive loss
        is approximated using pairwise distances within each mini-batch.
        Default is None (full-batch training).
    """  # noqa: E501

    _supports_mini_batch = True

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
        backend: Union[str, FaissConfig, None] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: float = 12.0,
        early_exaggeration_iter: Optional[int] = 250,
        max_iter_affinity: int = 100,
        metric: str = "sqeuclidean",
        sparsity: bool = True,
        check_interval: int = 50,
        compile: bool = False,
        encoder: Optional["torch.nn.Module"] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        self.metric = metric
        self.perplexity = perplexity
        self.lambda1 = lambda1
        self.gamma = gamma
        self.max_iter_affinity = max_iter_affinity
        self.sparsity = sparsity

        # Wrap encoder with Poincaré ball projection so outputs are
        # always valid hyperbolic points. RiemannianAdam handles both
        # ManifoldParameter (non-encoder) and regular params (encoder)
        # by falling back to Euclidean Adam for the latter.
        if encoder is not None:
            poincare = PoincareBallManifold()
            encoder = nn.Sequential(encoder, _ExpMap0(poincare))

        affinity_in = EntropicAffinity(
            perplexity=perplexity,
            metric=metric,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=sparsity,
        )
        super().__init__(
            affinity_in=affinity_in,
            affinity_out=None,
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
            encoder=encoder,
            batch_size=batch_size,
            **kwargs,
        )

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        # We compute once and for all the norms of X data samples
        self.X_norm = (X**2).sum(-1)
        return super()._fit_transform(X)

    def _compute_attractive_loss(self):
        distances_hyperbolic = pairwise_distances_indexed(
            self.embedding_,
            key_indices=self.NN_indices_,
            query_indices=self.chunk_indices_,
            metric="sqhyperbolic",
        )
        log_Q = (self.gamma / (distances_hyperbolic + self.gamma**2)).log()
        return cross_entropy_loss(self.affinity_in_, log_Q, log=True)

    def _compute_repulsive_loss(self):
        if self._use_mini_batch:
            embedding = self.embedding_[self.chunk_indices_]
            x_norm = self.X_norm[self.chunk_indices_]
        else:
            embedding = self.embedding_
            x_norm = self.X_norm
        distances_hyperbolic = pairwise_distances(
            embedding, metric="sqhyperbolic", backend=self.backend
        )
        log_Q = (self.gamma / (distances_hyperbolic + self.gamma**2)).log()
        rep_loss = logsumexp_red(log_Q, dim=(0, 1))
        Y_norm = (embedding**2).sum(-1)
        Y_norm = torch.arccosh(1 + 2 * (Y_norm / (1 - Y_norm)) + 1e-8) ** 2
        distance_term = ((x_norm - Y_norm) ** 2).mean()
        return rep_loss + self.lambda1 * distance_term
