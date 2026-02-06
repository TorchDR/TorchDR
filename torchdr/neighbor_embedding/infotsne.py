"""Info Noise-constrastive TSNE algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Dict, Union, Optional, Type
import torch

from torchdr.affinity import EntropicAffinity
from torchdr.neighbor_embedding.base import NegativeSamplingNeighborEmbedding
from torchdr.utils import logsumexp_red, cross_entropy_loss
from torchdr.distance import FaissConfig, pairwise_distances_indexed


class InfoTSNE(NegativeSamplingNeighborEmbedding):
    r"""InfoTSNE algorithm introduced in :cite:`damrich2022t`.

    It uses a :class:`~torchdr.EntropicAffinity` as input affinity :math:`\mathbf{P}`
    and a Student-t kernel as output affinity :math:`Q_{ij} = (1 + \| \mathbf{z}_i - \mathbf{z}_j \|^2)^{-1}`.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \sum_i \log \Big( \sum_{j \in \mathrm{Neg}(i)} Q_{ij} \Big)

    where :math:`\mathrm{Neg}(i)` is the set of negatives samples for point :math:`i`.

    Note
    ----
    This implementation supports multi-GPU training when launched with ``torchrun``.
    Set ``distributed='auto'`` (default) to automatically detect and use multiple GPUs.

    Parameters
    ----------
    perplexity : float
        Number of 'effective' nearest neighbors.
        Consider selecting a value between 2 and the number of samples.
        Different values can result in significantly different results.
    n_components : int, optional
        Dimension of the embedding space.
    lr : float or 'auto', optional
        Learning rate for the algorithm, by default "auto".
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto',
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is "LinearLR".
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    min_grad_norm : float, optional
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 1000.
    device : str, optional
        Device to use, by default "auto".
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
        Default is "faiss".
    verbose : bool, optional
        Verbosity, by default False.
    random_state : float, optional
        Random seed for reproducibility, by default None.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    early_exaggeration_coeff : float, optional
        Factor for the early exaggeration phase, by default 12.
    early_exaggeration_iter : int, optional
        Number of iterations for the early exaggeration phase, by default 250.
    n_negatives : int, optional
        Number of negative samples for the noise-contrastive loss, by default 300.
    sparsity : bool, optional
        Whether to use sparsity mode for the input affinity. Default is True.
    check_interval : int, optional
        Interval for checking convergence, by default 50.
    discard_NNs : bool, optional
        Whether to discard the nearest neighbors from the negative sampling.
        Default is False.
    compile : bool, optional
        Whether to compile the loss function with `torch.compile` for faster
        computation. Default is False.
    distributed : bool or 'auto', optional
        Whether to use distributed computation across multiple GPUs.
        - "auto": Automatically detect if running with torchrun (default)
        - True: Force distributed mode (requires torchrun)
        - False: Disable distributed mode
        Default is "auto".
    encoder : torch.nn.Module, optional
        A neural network that maps input data to the embedding space.
        When provided, optimizes encoder parameters instead of a raw
        embedding matrix. Default is None.
    batch_size : int, optional
        Mini-batch size for encoder-based training. When set with
        ``encoder``, each step processes a random subset through the
        encoder while using a cached full embedding for context.
        Default is None (full-batch).
    """  # noqa: E501

    def __init__(
        self,
        perplexity: float = 30,
        n_components: int = 2,
        lr: Union[float, str] = "auto",
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = "auto",
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = "LinearLR",
        scheduler_kwargs: Optional[Dict] = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        min_grad_norm: float = 1e-7,
        max_iter: int = 1000,
        device: str = "auto",
        backend: Union[str, FaissConfig, None] = "faiss",
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: Optional[float] = 12,
        early_exaggeration_iter: Optional[int] = 250,
        max_iter_affinity: int = 100,
        metric: str = "sqeuclidean",
        n_negatives: int = 300,
        sparsity: bool = True,
        check_interval: int = 50,
        discard_NNs: bool = False,
        compile: bool = False,
        distributed: Union[bool, str] = "auto",
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
            distributed=distributed,
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
            n_negatives=n_negatives,
            check_interval=check_interval,
            discard_NNs=discard_NNs,
            compile=compile,
            distributed=distributed,
            encoder=encoder,
            batch_size=batch_size,
            **kwargs,
        )

    def _compute_attractive_loss(self):
        distances_sq = pairwise_distances_indexed(
            self.embedding_,
            query_indices=self.chunk_indices_,
            key_indices=self.NN_indices_,
            metric="sqeuclidean",
        )
        log_Q = -(1 + distances_sq).log()
        return cross_entropy_loss(self.affinity_in_, log_Q, log=True)

    def _compute_repulsive_loss(self):
        distances_sq = pairwise_distances_indexed(
            self.embedding_,
            query_indices=self.chunk_indices_,
            key_indices=self.neg_indices_,
            metric="sqeuclidean",
        )
        log_Q = -(1 + distances_sq).log()
        return logsumexp_red(log_Q, dim=1).sum() / self.n_samples_in_
