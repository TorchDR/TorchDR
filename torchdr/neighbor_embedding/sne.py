"""Stochastic Neighbor embedding (SNE) algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Union, Type
import torch

from torchdr.affinity import EntropicAffinity
from torchdr.neighbor_embedding.base import SparseNeighborEmbedding
from torchdr.utils import logsumexp_red, cross_entropy_loss
from torchdr.distance import FaissConfig, pairwise_distances, pairwise_distances_indexed


class SNE(SparseNeighborEmbedding):
    r"""Stochastic Neighbor Embedding (SNE) introduced in :cite:`hinton2002stochastic`.

    It uses a :class:`~torchdr.EntropicAffinity` as input affinity :math:`\mathbf{P}`
    and a Gaussian kernel as output affinity :math:`Q_{ij} = \exp(- \| \mathbf{z}_i - \mathbf{z}_j \|^2)`.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \sum_i \log \Big( \sum_{j} Q_{ij} \Big) \:.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    n_components : int, optional
        Dimension of the embedded space (corresponds to the number of features of Z).
    lr : float or 'auto', optional
        Learning rate for the algorithm. By default 'auto'.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD". For best results, we recommend using "SGD" with 'auto' learning rate.
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto',
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Arguments for the scheduler.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z.
    init_scaling : float, optional
        Scaling factor for the initialization.
    min_grad_norm : float, optional
        Precision threshold at which the algorithm stops.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm.
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
        By default 10.0 for early exaggeration.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default 250.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    sparsity : bool, optional
        Whether to use sparsity in the algorithm.
    check_interval : int, optional
        Interval for checking the convergence of the algorithm.
    compile : bool, optional
        Whether to compile the algorithm using torch.compile. Default is False.
    encoder : torch.nn.Module, optional
        A neural network that maps input data to the embedding space.
        When provided, enables out-of-sample extension via ``transform(X_new)``.
        Default is None.
    batch_size : int, optional
        Mini-batch size for encoder-based training. The repulsive partition
        function is approximated using pairwise distances within each
        mini-batch. Default is None (full-batch training).
    """  # noqa: E501

    _supports_mini_batch = True

    def __init__(
        self,
        perplexity: float = 30,
        n_components: int = 2,
        lr: Union[float, str] = "auto",
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = "auto",
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
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
        )
        super().__init__(
            affinity_in=affinity_in,
            affinity_out=None,
            n_components=n_components,
            optimizer=optimizer,
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

    def _compute_attractive_loss(self):
        distances_sq = pairwise_distances_indexed(
            self.embedding_,
            key_indices=self.NN_indices_,
            query_indices=self.chunk_indices_,
            metric="sqeuclidean",
        )
        return cross_entropy_loss(self.affinity_in_, -distances_sq, log=True)

    def _compute_repulsive_loss(self):
        if self._use_mini_batch:
            embedding = self.embedding_[self.chunk_indices_]
            n = len(self.chunk_indices_)
        else:
            embedding = self.embedding_
            n = self.n_samples_in_
        distances_sq = pairwise_distances(
            embedding, metric="sqeuclidean", backend=self.backend
        )
        return logsumexp_red(-distances_sq, dim=1).sum() / n
