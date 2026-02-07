"""SNEkhorn algorithm (inverse OT DR)."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import warnings
from typing import Dict, Optional, Union, Type
import torch

from torchdr.affinity import (
    EntropicAffinity,
    SinkhornAffinity,
    SymmetricEntropicAffinity,
)
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.distance import FaissConfig
from torchdr.utils import cross_entropy_loss, logsumexp_red, bool_arg


class TSNEkhorn(AffinityMatcher):
    r"""TSNEkhorn algorithm introduced in :cite:`van2024snekhorn`.

    It uses a :class:`~torchdr.SymmetricEntropicAffinity` as input affinity :math:`\mathbf{P}`
    and a :class:`~torchdr.SinkhornAffinity` (doubly stochastic normalization of the Student-t kernel) as output affinity :math:`\mathbf{Q}`.

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
    lr : float or 'auto', optional
        Learning rate for the algorithm. By default 'auto', which sets the
        learning rate based on the number of samples.
    optimizer : str or torch.optim.Optimizer, optional
        Name of an optimizer from torch.optim or an optimizer class.
        Default is "SGD".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer. Default is 'auto',
        which sets momentum to 0.8 for SGD.
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
    lr_affinity_in : float, optional
        Learning rate used to update dual variables for the symmetric entropic
        affinity computation.
    eps_square_affinity_in : bool, optional
        When computing the symmetric entropic affinity, whether to optimize
        on the square of the dual variables. May be more stable in practice.
    tol_affinity_in : float, optional
        Precision threshold for the symmetric entropic affinity computation.
    max_iter_affinity_in : int, optional
        Number of maximum iterations for the symmetric entropic affinity computation.
    metric : {'sqeuclidean', 'manhattan'}, optional
        Metric to use for the input affinity, by default 'sqeuclidean'.
    unrolling : bool, optional
        Whether to use unrolling for solving inverse OT. If False, uses
        the gap objective. Default is False.
    symmetric_affinity : bool, optional
        Whether to use symmetric entropic affinity. If False, uses
        entropic affinity. Default is True.
    check_interval : int, optional
        Interval for checking the convergence of the algorithm, by default 50.
    compile : bool, optional
        Whether to compile the algorithm using torch.compile. Default is False.
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
        min_grad_norm: float = 1e-4,
        max_iter: int = 2000,
        device: str = "auto",
        backend: Union[str, FaissConfig, None] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        lr_affinity_in: float = 1e-1,
        eps_square_affinity_in: bool = True,
        tol_affinity_in: float = 1e-3,
        max_iter_affinity_in: int = 100,
        metric: str = "sqeuclidean",
        unrolling: bool = False,
        symmetric_affinity: bool = True,
        check_interval: int = 50,
        compile: bool = False,
        **kwargs,
    ):
        self.metric = metric
        self.perplexity = perplexity
        self.lr_affinity_in = lr_affinity_in
        self.eps_square_affinity_in = bool_arg(eps_square_affinity_in)
        self.max_iter_affinity_in = max_iter_affinity_in
        self.tol_affinity_in = tol_affinity_in
        self.unrolling = bool_arg(unrolling)
        self.symmetric_affinity = bool_arg(symmetric_affinity)

        if self.symmetric_affinity:
            affinity_in = SymmetricEntropicAffinity(
                perplexity=perplexity,
                lr=lr_affinity_in,
                eps_square=eps_square_affinity_in,
                metric=metric,
                tol=tol_affinity_in,
                max_iter=max_iter_affinity_in,
                device=device,
                backend=backend,
                verbose=verbose,
                zero_diag=False,
            )
        else:
            affinity_in = EntropicAffinity(
                perplexity=perplexity,
                metric=metric,
                max_iter=max_iter_affinity_in,
                device=device,
                backend=backend,
                verbose=verbose,
            )
        affinity_out = SinkhornAffinity(
            metric="sqeuclidean",
            device=device,
            backend=backend,
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
            **kwargs,
        )

    def _fit_transform(self, X: torch.Tensor, y=None) -> torch.Tensor:
        n_samples = X.shape[0]
        if n_samples <= self.perplexity:
            raise ValueError(
                f"[TorchDR] ERROR : Number of samples is smaller than perplexity "
                f"({n_samples} <= {self.perplexity})."
            )
        return super()._fit_transform(X, y)

    def _set_learning_rate(self):
        if self.lr == "auto":
            if self.optimizer != "SGD":
                if self.verbose:
                    warnings.warn(
                        "[TorchDR] WARNING : when 'auto' is used for the learning "
                        "rate, the optimizer should be 'SGD'."
                    )
            self.lr_ = max(self.n_samples_in_ / 4, 50)
        else:
            self.lr_ = self.lr

    def _configure_optimizer(self):
        if isinstance(self.optimizer, str):
            try:
                optimizer_class = getattr(torch.optim, self.optimizer)
            except AttributeError:
                raise ValueError(
                    f"[TorchDR] ERROR: Optimizer '{self.optimizer}' not found in torch.optim"
                )
        else:
            if not issubclass(self.optimizer, torch.optim.Optimizer):
                raise ValueError(
                    "[TorchDR] ERROR: optimizer must be a string (name of an optimizer in "
                    "torch.optim) or a subclass of torch.optim.Optimizer"
                )
            optimizer_class = self.optimizer

        if self.optimizer_kwargs == "auto":
            if self.optimizer == "SGD":
                optimizer_kwargs = {"momentum": 0.8}
            else:
                optimizer_kwargs = {}
        else:
            optimizer_kwargs = self.optimizer_kwargs or {}

        self.optimizer_ = optimizer_class(self.params_, lr=self.lr_, **optimizer_kwargs)
        return self.optimizer_

    def _compute_loss(self):
        if not hasattr(self, "dual_sinkhorn_"):
            self.dual_sinkhorn_ = None

        log_Q = self.affinity_out(
            self.embedding_, log=True, init_dual=self.dual_sinkhorn_
        )
        dual_sinkhorn = self.affinity_out.dual_.detach()
        if hasattr(self, "dual_sinkhorn_"):
            self.dual_sinkhorn_ = dual_sinkhorn
        else:
            self.register_buffer("dual_sinkhorn_", dual_sinkhorn, persistent=False)

        attractive_term = cross_entropy_loss(self.affinity_in_, log_Q, log=True)
        if self.unrolling:
            repulsive_term = 0
        else:
            repulsive_term = logsumexp_red(log_Q, dim=(0, 1)).exp()

        loss = attractive_term + repulsive_term
        return loss
