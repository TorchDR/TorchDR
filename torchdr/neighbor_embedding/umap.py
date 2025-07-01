"""UMAP algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Union, Type
import torch

from torchdr.affinity import UMAPAffinityIn, UMAPAffinityOut
from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from torchdr.utils import cross_entropy_loss, sum_red


class UMAP(SampledNeighborEmbedding):
    r"""UMAP introduced in :cite:`mcinnes2018umap` and further studied in :cite:`damrich2021umap`.

    It uses a :class:`~torchdr.UMAPAffinityIn` as input
    affinity :math:`\mathbf{P}` and a :class:`~torchdr.UMAPAffinityOut` as output
    affinity :math:`\mathbf{Q}`.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \sum_{i,j \in \mathrm{Neg}(i)} \log (1 - Q_{ij})

    where :math:`\mathrm{Neg}(i)` is the set of negatives samples for point :math:`i`.

    Parameters
    ----------
    n_neighbors : float, optional
        Number of nearest neighbors.
    n_components : int, optional
        Dimension of the embedding space.
    min_dist : float, optional
        Minimum distance between points in the embedding space.
    spread : float, optional
        The effective scale of the embedded points. Used to configure the UMAPAffinityOut.
    a : float, optional
        Parameter for the Student t-distribution.
    b : float, optional
        Parameter for the Student t-distribution.
    lr : float, optional
        Learning rate for the algorithm, by default 1e-1.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto'.
        which sets appropriate momentum values for SGD based on early exaggeration phase.
    scheduler : str or torch.optim.lr_scheduler.LRScheduler, optional
        Name of a scheduler from torch.optim.lr_scheduler or a scheduler class.
        Default is "LinearLR".
    scheduler_kwargs : dict, 'auto', or None, optional
        Additional keyword arguments for the scheduler. Default is 'auto', which
        corresponds to a linear decay from the learning rate to 0 for `LinearLR`.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    min_grad_norm : float, optional
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm. by default 2000.
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
        Precision threshold for the input affinity computation.
    max_iter_affinity : int, optional
        Number of maximum iterations for the input affinity computation.
    metric_in : {'euclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'euclidean'.
    metric_out : {'euclidean', 'manhattan'}, optional
        Metric to use for the output affinity, by default 'euclidean'.
    n_negatives : int, optional
        Number of negative samples for the noise-contrastive loss, by default 10.
    sparsity : bool, optional
        Whether to use sparsity mode for the input affinity. Default is True.
    check_interval : int, optional
        Check interval for the algorithm, by default 50.
    discard_NNs : bool, optional
        Whether to discard the nearest neighbors from the negative sampling.
        Default is False.
    compile : bool, optional
        Whether to compile the algorithm using torch.compile. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 30,
        n_components: int = 2,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        lr: float = 1e-1,
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "SGD",
        optimizer_kwargs: Union[Dict, str] = "auto",
        scheduler: Optional[
            Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]
        ] = "LinearLR",
        scheduler_kwargs: Union[Dict, str, None] = "auto",
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
        n_negatives: int = 10,
        sparsity: bool = True,
        check_interval: int = 50,
        discard_NNs: bool = False,
        compile: bool = False,
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.a = a
        self.b = b
        self.metric_in = metric_in
        self.metric_out = metric_out
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity
        self.sparsity = sparsity

        affinity_in = UMAPAffinityIn(
            n_neighbors=n_neighbors,
            metric=metric_in,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=sparsity,
        )
        affinity_out = UMAPAffinityOut(
            min_dist=min_dist,
            spread=spread,
            a=a,
            b=b,
            metric=metric_out,
            device=device,
            verbose=False,
        )

        _scheduler_kwargs = scheduler_kwargs
        if scheduler == "LinearLR" and scheduler_kwargs == "auto":
            _scheduler_kwargs = {
                "start_factor": 1.0,
                "end_factor": 0,
                "total_iters": max_iter,
            }

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
            scheduler_kwargs=_scheduler_kwargs,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            n_negatives=n_negatives,
            check_interval=check_interval,
            discard_NNs=discard_NNs,
            compile=compile,
            **kwargs,
        )

    def _repulsive_loss(self):
        Q = self.affinity_out(self.embedding_, indices=self.neg_indices_)
        Q = Q / (Q + 1)  # stabilization trick, PR #856 from UMAP repo
        return -sum_red((1 - Q).log(), dim=(0, 1))

    def _attractive_loss(self):
        Q = self.affinity_out(self.embedding_, indices=self.NN_indices_)
        Q = Q / (Q + 1)  # stabilization trick, PR #856 from UMAP repo
        return cross_entropy_loss(self.affinity_in_, Q)
