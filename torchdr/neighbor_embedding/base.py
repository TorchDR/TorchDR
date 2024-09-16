# -*- coding: utf-8 -*-
"""Base classes for Neighbor Embedding methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import warnings

from torchdr.affinity import (
    Affinity,
    UnnormalizedAffinity,
    UnnormalizedLogAffinity,
    SparseLogAffinity,
)
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.utils import cross_entropy_loss, OPTIMIZERS


class NeighborEmbedding(AffinityMatcher):
    r"""Solves the neighbor embedding problem.

    It amounts to solving:

    .. math::

        \min_{\mathbf{Z}} \: - \lambda \sum_{ij} P_{ij} \log Q_{ij} + \gamma \mathcal{L}_{\mathrm{rep}}( \mathbf{Q})

    where :math:`\mathbf{P}` is the input affinity matrix, :math:`\mathbf{Q}` is the
    output affinity matrix, :math:`\mathcal{L}_{\mathrm{rep}}` is the repulsive
    term of the loss function, :math:`\lambda` is the :attr:`early_exaggeration`
    parameter and :math:`\gamma` is the :attr:`coeff_repulsion` parameter.

    Note that the early exaggeration coefficient :math:`\lambda` is set to
    :math:`1` after the early exaggeration phase which duration is controlled by the
    :attr:`early_exaggeration_iter` parameter.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity
        The affinity object for the output embedding space.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    optimizer : str or 'auto', optional
        Optimizer to use for the optimization. Default is "Adam".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer.
    scheduler : str, optional
        Learning rate scheduler. Default is "constant".
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 2000.
    init : str, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    tolog : bool, optional
        If True, logs the optimization process. Default is False.
    device : str, optional
        Device to use for computations. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for computations. Default is False.
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is 0.
    early_exaggeration : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        Default is 1.0.
    coeff_repulsion : float, optional
        Coefficient for the repulsion term. Default is 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration. Default is None.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Affinity,
        kwargs_affinity_out: dict = {},
        n_components: int = 2,
        lr: float | str = 1e0,
        optimizer: str = "Adam",
        optimizer_kwargs: dict | str = None,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        tol: float = 1e-7,
        max_iter: int = 2000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        tolog: bool = False,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
        random_state: float = 0,
        early_exaggeration: float = 1.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = None,
        **kwargs,
    ):
        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            kwargs_affinity_out=kwargs_affinity_out,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            tol=tol,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
        )

        self.early_exaggeration = early_exaggeration
        self.coeff_repulsion = coeff_repulsion
        self.early_exaggeration_iter = early_exaggeration_iter

        # improve consistency with the sklearn API
        if "learning_rate" in kwargs:
            self.lr = kwargs["learning_rate"]
        if "min_grad_norm" in kwargs:
            self.tol = kwargs["min_grad_norm"]
        if "early_exaggeration" in kwargs:
            self.early_exaggeration = kwargs["early_exaggeration"]

    def _additional_updates(self, step):
        if (  # stop early exaggeration phase
            self.early_exaggeration_ > 1 and step == self.early_exaggeration_iter
        ):
            self.early_exaggeration_ = 1
            # reinitialize optim
            self._set_learning_rate()
            self._set_optimizer()
            self._set_scheduler()

        return self

    def _check_n_neighbors(self, n):
        param_list = ["perplexity", "n_neighbors"]

        for param_name in param_list:
            if hasattr(self, param_name):
                param_value = getattr(self, param_name)
                if n <= param_value:
                    if self.verbose:
                        warnings.warn(
                            "[TorchDR] WARNING : Number of samples is smaller than "
                            f"{param_name} ({n} <= {param_value}), setting "
                            f"{param_name} to {n//2} (which corresponds to n//2)."
                        )
                    new_value = n // 2
                    setattr(self, param_name + "_", new_value)
                    setattr(self.affinity_in, param_name, new_value)

        return self

    def _fit(self, X: torch.Tensor):
        self._check_n_neighbors(X.shape[0])
        self.early_exaggeration_ = (
            self.early_exaggeration
        )  # early_exaggeration_ may change during the optimization

        super()._fit(X)

    def _loss(self):
        raise NotImplementedError("[TorchDR] ERROR : _loss method must be implemented.")

    def _set_learning_rate(self):
        if self.lr == "auto":
            if self.optimizer not in ["auto", "SGD"]:
                if self.verbose:
                    warnings.warn(
                        "[TorchDR] WARNING : when 'auto' is used for the learning "
                        "rate, the optimizer should be 'SGD'."
                    )
            # from the sklearn TSNE implementation
            self.lr_ = max(self.n_samples_in_ / self.early_exaggeration_ / 4, 50)
        else:
            self.lr_ = self.lr

    def _set_optimizer(self):
        optimizer = "SGD" if self.optimizer == "auto" else self.optimizer
        # from the sklearn TSNE implementation
        if self.optimizer_kwargs == "auto":
            if self.optimizer == "SGD":
                if self.early_exaggeration_ > 1:
                    optimizer_kwargs = {"momentum": 0.5}
                else:
                    optimizer_kwargs = {"momentum": 0.8}
            else:
                optimizer_kwargs = {}
        else:
            optimizer_kwargs = self.optimizer_kwargs

        self.optimizer_ = OPTIMIZERS[optimizer](
            self.params_, lr=self.lr_, **(optimizer_kwargs or {})
        )
        return self.optimizer_

    def _set_scheduler(self):
        if self.early_exaggeration_ > 1:
            n_iter = min(self.early_exaggeration_iter, self.max_iter)
        else:
            n_iter = self.max_iter - self.early_exaggeration_iter
        super()._set_scheduler(n_iter)


class SparseNeighborEmbedding(NeighborEmbedding):
    r"""Solves the neighbor embedding problem with a sparse input affinity matrix.

    It amounts to solving:

    .. math::

        \min_{\mathbf{Z}} \: - \lambda \sum_{ij} P_{ij} \log Q_{ij} + \gamma \mathcal{L}_{\mathrm{rep}}( \mathbf{Q})

    where :math:`\mathbf{P}` is the input affinity matrix, :math:`\mathbf{Q}` is the
    output affinity matrix, :math:`\mathcal{L}_{\mathrm{rep}}` is the repulsive
    term of the loss function, :math:`\lambda` is the :attr:`early_exaggeration`
    parameter and :math:`\gamma` is the :attr:`coeff_repulsion` parameter.

    **Fast attraction.** This class should be used when the input affinity matrix is a :class:`~torchdr.SparseLogAffinity` and the output affinity matrix is an :class:`~torchdr.UnnormalizedAffinity`. In such cases, the attractive term can be computed with linear complexity.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity
        The affinity object for the output embedding space.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    optimizer : str or 'auto', optional
        Optimizer to use for the optimization. Default is "Adam".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer.
    scheduler : str, optional
        Learning rate scheduler. Default is "constant".
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 2000.
    init : str, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    tolog : bool, optional
        If True, logs the optimization process. Default is False.
    device : str, optional
        Device to use for computations. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for computations. Default is False.
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is 0.
    early_exaggeration : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        Default is 1.0.
    coeff_repulsion : float, optional
        Coefficient for the repulsion term. Default is 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration. Default is None.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Affinity,
        kwargs_affinity_out: dict = {},
        n_components: int = 2,
        lr: float | str = 1e0,
        optimizer: str = "Adam",
        optimizer_kwargs: dict | str = None,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        tol: float = 1e-7,
        max_iter: int = 2000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        tolog: bool = False,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
        random_state: float = 0,
        early_exaggeration: float = 1.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = None,
    ):
        # check affinity affinity_in
        if not isinstance(affinity_in, SparseLogAffinity):
            raise NotImplementedError(
                "[TorchDR] ERROR : when using SparseNeighborEmbedding, affinity_in "
                "must be a sparse affinity."
            )

        # check affinity affinity_out
        if not isinstance(affinity_out, UnnormalizedAffinity):
            raise NotImplementedError(
                "[TorchDR] ERROR : when using SparseNeighborEmbedding, affinity_out "
                "must be an UnnormalizedAffinity object."
            )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            kwargs_affinity_out=kwargs_affinity_out,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            tol=tol,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
            early_exaggeration=early_exaggeration,
            coeff_repulsion=coeff_repulsion,
            early_exaggeration_iter=early_exaggeration_iter,
        )

    def _attractive_loss(self):
        if isinstance(self.affinity_out, UnnormalizedLogAffinity):
            log_Q = self.affinity_out(
                self.embedding_, log=True, indices=self.NN_indices_
            )
            return cross_entropy_loss(self.PX_, log_Q, log=True)
        else:
            Q = self.affinity_out(self.embedding_, indices=self.NN_indices_)
            return cross_entropy_loss(self.PX_, Q)

    def _repulsive_loss(self):
        raise NotImplementedError(
            "[TorchDR] ERROR : _repulsive_loss method must be implemented."
        )

    def _loss(self):
        loss = (
            self.early_exaggeration_ * self._attractive_loss()
            + self.coeff_repulsion * self._repulsive_loss()
        )
        return loss


class SampledNeighborEmbedding(SparseNeighborEmbedding):
    r"""Solves the neighbor embedding problem with both sparsity and sampling.

    It amounts to solving:

    .. math::

        \min_{\mathbf{Z}} \: - \lambda \sum_{ij} P_{ij} \log Q_{ij} + \gamma \mathcal{L}_{\mathrm{rep}}( \mathbf{Q})

    where :math:`\mathbf{P}` is the input affinity matrix, :math:`\mathbf{Q}` is the
    output affinity matrix, :math:`\mathcal{L}_{\mathrm{rep}}` is the repulsive
    term of the loss function, :math:`\lambda` is the :attr:`early_exaggeration`
    parameter and :math:`\gamma` is the :attr:`coeff_repulsion` parameter.

    **Fast attraction.** This class should be used when the input affinity matrix is a
    :class:`~torchdr.SparseLogAffinity` and the output affinity matrix is an
    :class:`~torchdr.UnnormalizedAffinity`. In such cases, the attractive term
    can be computed with linear complexity.

    **Fast repulsion.** A stochastic estimation of the repulsive term is used
    to reduce its complexity to linear.
    This is done by sampling a fixed number of negative samples
    :attr:`n_negatives` for each point.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity
        The affinity object for the output embedding space.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    optimizer : str or 'auto', optional
        Optimizer to use for the optimization. Default is "Adam".
    optimizer_kwargs : dict or 'auto', optional
        Additional keyword arguments for the optimizer.
    scheduler : str, optional
        Learning rate scheduler. Default is "constant".
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 2000.
    init : str, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    tolog : bool, optional
        If True, logs the optimization process. Default is False.
    device : str, optional
        Device to use for computations. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for computations. Default is False.
    verbose : bool, optional
        Verbosity of the optimization process. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is 0.
    early_exaggeration : float, optional
        Coefficient for the attraction term during the early exaggeration phase.
        Default is 1.0.
    coeff_repulsion : float, optional
        Coefficient for the repulsion term. Default is 1.0.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration. Default is None.
    n_negatives : int, optional
        Number of negative samples for the repulsive loss.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Affinity,
        kwargs_affinity_out: dict = {},
        n_components: int = 2,
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        lr: float | str = 1e0,
        scheduler: str = "constant",
        scheduler_kwargs: dict | str = None,
        tol: float = 1e-7,
        max_iter: int = 2000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        tolog: bool = False,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
        random_state: float = 0,
        early_exaggeration: float = 1.0,
        coeff_repulsion: float = 1.0,
        early_exaggeration_iter: int = None,
        n_negatives: int = 5,
    ):
        self.n_negatives = n_negatives

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            kwargs_affinity_out=kwargs_affinity_out,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            tol=tol,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
            early_exaggeration=early_exaggeration,
            coeff_repulsion=coeff_repulsion,
            early_exaggeration_iter=early_exaggeration_iter,
        )

    def _sample_negatives(self, discard_NNs=True):
        # Negatives are all other points except NNs (if discard_NNs) and point itself
        n_possible_negatives = self.n_samples_in_ - 1  # Exclude the self-index
        discard_NNs_ = discard_NNs and self.NN_indices_ is not None
        if discard_NNs_:
            n_possible_negatives -= self.NN_indices_.shape[-1]  # Exclude the NNs

        if not hasattr(self, "n_negatives_"):
            if self.n_negatives > n_possible_negatives:
                if self.verbose:
                    warnings.warn(
                        "[TorchDR] WARNING: n_negatives is too large. "
                        "Setting n_negatives to the difference between the number of "
                        "samples and the number of neighbors."
                    )
                self.n_negatives_ = n_possible_negatives
            else:
                self.n_negatives_ = self.n_negatives

        indices = self.generator_.integers(
            1, n_possible_negatives, (self.n_samples_in_, self.n_negatives_)
        )
        device = getattr(self.NN_indices_, "device", "cpu")
        indices = torch.from_numpy(indices).to(device=device)

        exclude_indices = (
            torch.arange(self.n_samples_in_).unsqueeze(1).to(device=device)
        )  # Self indices
        if discard_NNs_:
            exclude_indices = torch.cat(
                (
                    exclude_indices,
                    self.NN_indices_,
                ),  # Concatenate self and NNs
                dim=1,
            )

        # Adjusts sampled indices to take into account excluded indices
        exclude_indices.sort(axis=1)
        adjustments = torch.searchsorted(exclude_indices, indices, right=True)
        indices += adjustments

        return indices
