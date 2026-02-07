# -*- coding: utf-8 -*-
"""Hyperbolic Stochastic Neighbor Embedding (CO-SNE) algorithm."""

# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Union, Type, Any
import torch

from torchdr.neighbor_embedding.base import NeighborEmbedding
from torchdr.affinity import EntropicAffinity
from torchdr.distance import FaissConfig, pairwise_distances, pairwise_distances_indexed
from torchdr.utils import logsumexp_red, RiemannianAdam, cross_entropy_loss


class COSNE(NeighborEmbedding):
    r"""Implementation of the CO-Stochastic Neighbor Embedding (CO-SNE) introduced in :cite:`guo2022co`.

    This algorithm is a variant of SNE that uses a hyperbolic space for the embedding.
    It uses a :class:`~torchdr.EntropicAffinity` as input affinity :math:`\mathbf{P}`
    and a Cauchy kernel in hyperbolic space as output affinity :math:`Q_{ij} = \gamma / (d_H(\mathbf{z}_i, \mathbf{z}_j) + \gamma^2)` where :math:`d_H` is the hyperbolic distance.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \log \Big( \sum_{ij} Q_{ij} \Big) + \lambda_1 \cdot \frac{1}{n} \sum_i \Big( \| \mathbf{x}_i \|^2 - d_H(\mathbf{z}_i, \mathbf{0})^2 \Big)^2

    where the first two terms form the KL divergence between :math:`\mathbf{P}` and
    :math:`\mathbf{Q}` (up to a constant) and the third term regularizes the embedding
    to preserve the norms of the input data in hyperbolic space.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    learning_rate_for_h_loss : float
        Coefficient for the distance preservation loss enforcing that the
        hyperbolic distance to the origin of each embedded point matches
        the squared Euclidean norm of the corresponding input sample.
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
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    sparsity : bool, optional
        Whether to use sparsity mode for the input affinity. Default is True.
    check_interval : int, optional
        Number of iterations between checks for convergence, by default 50.
    distributed : bool or 'auto', optional
        Whether to use distributed computation across multiple GPUs.
        - "auto": Automatically detect if running with torchrun (default)
        - True: Force distributed mode (requires torchrun)
        - False: Disable distributed mode
        Default is "auto".
    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        learning_rate_for_h_loss: float = 1,
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
        max_iter_affinity: int = 100,
        metric: str = "sqeuclidean",
        sparsity: bool = True,
        check_interval: int = 50,
        compile: bool = False,
        distributed: Union[bool, str] = "auto",
        **kwargs,
    ):
        self.metric = metric
        self.perplexity = perplexity
        self.learning_rate_for_h_loss = learning_rate_for_h_loss
        self.gamma = gamma
        self.max_iter_affinity = max_iter_affinity
        self.sparsity = sparsity

        affinity_in = EntropicAffinity(
            perplexity=perplexity,
            metric=metric,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=sparsity,
            distributed=distributed,
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
            check_interval=check_interval,
            compile=compile,
            distributed=distributed,
            **kwargs,
        )

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        # We compute once and for all the norms of X data samples
        self.X_norm = (X**2).sum(-1)
        return super()._fit_transform(X)

    def _compute_attractive_loss(self):
        distances_hyperbolic = pairwise_distances_indexed(
            self.embedding_,
            query_indices=self.chunk_indices_,
            key_indices=self.NN_indices_,
            metric="sqhyperbolic",
        )
        log_Q = (self.gamma / (distances_hyperbolic + self.gamma**2)).log()
        return cross_entropy_loss(self.affinity_in_, log_Q, log=True)

    def _compute_repulsive_loss(self):
        distances_hyperbolic = pairwise_distances(
            self.embedding_, metric="sqhyperbolic", backend=self.backend
        )
        log_Q = (self.gamma / (distances_hyperbolic + self.gamma**2)).log()
        rep_loss = logsumexp_red(log_Q, dim=(0, 1))

        # Distance preservation: hyperbolic distance to origin should match
        # the squared Euclidean norm of the input data.
        Y_norm = (self.embedding_**2).sum(-1)
        Y_norm = torch.arccosh(1 + 2 * (Y_norm / (1 - Y_norm)) + 1e-8) ** 2
        distance_term = ((self.X_norm.to(Y_norm.device) - Y_norm) ** 2).mean()

        loss = rep_loss + self.learning_rate_for_h_loss * distance_term
        if getattr(self, "world_size", 1) > 1:
            loss = loss / self.world_size
        return loss
