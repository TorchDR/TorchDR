"""Info Noise-constrastive TSNE algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Dict, Union, Optional, Type
import torch

from torchdr.affinity import EntropicAffinity, StudentAffinity
from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from torchdr.utils import logsumexp_red


class InfoTSNE(SampledNeighborEmbedding):
    r"""InfoTSNE algorithm introduced in :cite:`damrich2022t`.

    It uses a :class:`~torchdr.EntropicAffinity` as input
    affinity :math:`\mathbf{P}` and a :class:`~torchdr.StudentAffinity` as output
    affinity :math:`\mathbf{Q}`.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \sum_i \log \Big( \sum_{j \in \mathrm{Neg}(i)} Q_{ij} \Big)

    where :math:`\mathrm{Neg}(i)` is the set of negatives samples for point :math:`i`.

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
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto',
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is None (no scheduler).
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    min_grad_norm : float, optional
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 2000.
    device : str, optional
        Device to use, by default "auto".
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is "faiss".
    verbose : bool, optional
        Verbosity, by default False.
    random_state : float, optional
        Random seed for reproducibility, by default None.
    tol_affinity : float, optional
        Precision threshold for the entropic affinity root search.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric_in : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    metric_out : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the output affinity, by default 'sqeuclidean'.
    n_negatives : int, optional
        Number of negative samples for the noise-contrastive loss, by default 5.
    sparsity : bool, optional
        Whether to use sparsity mode for the input affinity. Default is True.
    check_interval : int, optional
        Interval for checking convergence, by default 50.
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
        ] = None,
        scheduler_kwargs: Optional[Dict] = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        min_grad_norm: float = 1e-7,
        max_iter: int = 2000,
        device: Optional[str] = None,
        backend: Optional[str] = "faiss",
        verbose: bool = False,
        random_state: Optional[float] = None,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "sqeuclidean",
        metric_out: str = "sqeuclidean",
        n_negatives: int = 50,
        sparsity: bool = True,
        check_interval: int = 50,
    ):
        self.metric_in = metric_in
        self.metric_out = metric_out
        self.perplexity = perplexity
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity
        self.sparsity = sparsity

        affinity_in = EntropicAffinity(
            perplexity=perplexity,
            metric=metric_in,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=sparsity,
        )
        affinity_out = StudentAffinity(
            metric=metric_out,
            device=device,
            verbose=False,
        )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
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
            n_negatives=n_negatives,
            check_interval=check_interval,
        )

    def _repulsive_loss(self):
        indices = self._sample_negatives()
        log_Q = self.affinity_out(self.embedding_, log=True, indices=indices)
        return logsumexp_red(log_Q, dim=1).sum() / self.n_samples_in_
