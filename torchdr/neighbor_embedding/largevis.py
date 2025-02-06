"""LargeVis algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.affinity import EntropicAffinity, StudentAffinity
from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from torchdr.utils import cross_entropy_loss, sum_output


class LargeVis(SampledNeighborEmbedding):
    r"""LargeVis algorithm introduced in :cite:`tang2016visualizing`.

    It uses a :class:`~torchdr.EntropicAffinity` as input
    affinity :math:`\mathbf{P}` and a :class:`~torchdr.StudentAffinity` as output
    affinity :math:`\mathbf{Q}`.

    The loss function is defined as:

    .. math::

        -\sum_{ij} P_{ij} \log Q_{ij} + \sum_{i,j \in \mathrm{Neg}(i)} \log (1 - Q_{ij})

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
    optimizer : {'SGD', 'Adam', 'NAdam', 'auto}, optional
        Which pytorch optimizer to use, by default 'auto'.
    optimizer_kwargs : dict or 'auto, optional
        Arguments for the optimizer, by default 'auto'.
    scheduler : {'constant', 'linear'}, optional
        Learning rate scheduler.
    scheduler_kwargs : dict, optional
        Arguments for the scheduler, by default None.
    init : {'normal', 'pca'} or torch.Tensor of shape (n_samples, output_dim), optional
        Initialization for the embedding Z, default 'pca'.
    init_scaling : float, optional
        Scaling factor for the initialization, by default 1e-4.
    min_grad_norm : float, optional
        Precision threshold at which the algorithm stops, by default 1e-7.
    max_iter : int, optional
        Number of maximum iterations for the descent algorithm, by default 3000.
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
    tol_affinity : _type_, optional
        Precision threshold for the entropic affinity root search.
    max_iter_affinity : int, optional
        Number of maximum iterations for the entropic affinity root search.
    metric_in : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    metric_out : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the output affinity, by default 'sqeuclidean'.
    n_negatives : int, optional
        Number of negative samples for the repulsive loss.
    sparsity : bool, optional
        Whether to use sparsity mode for the input affinity. Default is True.
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
        min_grad_norm: float = 1e-7,
        max_iter: int = 3000,
        device: str = None,
        backend: str = None,
        verbose: bool = False,
        random_state: float = None,
        early_exaggeration_coeff: float = 12.0,
        early_exaggeration_iter: int = 250,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "sqeuclidean",
        metric_out: str = "sqeuclidean",
        n_negatives: int = 5,
        sparsity: bool = True,
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
            backend=backend,
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
            early_exaggeration_coeff=early_exaggeration_coeff,
            early_exaggeration_iter=early_exaggeration_iter,
            n_negatives=n_negatives,
        )

    @sum_output
    def _repulsive_loss(self):
        indices = self._sample_negatives()
        Q = self.affinity_out(self.embedding_, indices=indices)
        Q = Q / (Q + 1)
        return -(1 - Q).log() / self.n_samples_in_

    def _attractive_loss(self):
        Q = self.affinity_out(self.embedding_, indices=self.NN_indices_)
        Q = Q / (Q + 1)
        return cross_entropy_loss(self.PX_, Q)
