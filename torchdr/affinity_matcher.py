# -*- coding: utf-8 -*-
"""
Affinity matcher base classes
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from tqdm import tqdm

from torchdr.utils import (
    OPTIMIZERS,
    check_nonnegativity,
    check_NaNs,
    handle_backend,
)
from torchdr.affinity import Affinity
from torchdr.spectral import PCA
from torchdr.base import DRModule
from torchdr.losses import square_loss, cross_entropy_loss, binary_cross_entropy_loss

LOSS_DICT = {
    "square_loss": square_loss,
    "cross_entropy_loss": cross_entropy_loss,
    "binary_cross_entropy_loss": binary_cross_entropy_loss,
}


class AffinityMatcher(DRModule):
    r"""
    Performs dimensionality reduction by matching two affinity matrices.
    It amounts to solving the following optimization problem:

    .. math::

        \min_{\mathbf{Z}} \: \sum_{ij} L( [\mathbf{A_X}]_{ij}, [\mathbf{A_Z}]_{ij}) \:.

    Optimization of the embedding is perfomed using torch autodiff.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity
        The affinity object for the output embedding space.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out fit_transform method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    optimizer : str, optional
        Optimizer to use for the optimization. Default is "Adam".
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e0.
    scheduler : str, optional
        Learning rate scheduler. Default is "constant".
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    init : str, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    early_exaggeration : int, optional
        Early exaggeration factor, by default None.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default None.
    tolog : bool, optional
        If True, logs the optimization process. Default is False.
    device : str, optional
        Device to use for computations. Default is None.
    keops : bool, optional
        Whether to use KeOps for computations. Default is True.
    verbose : bool, optional
        Verbosity of the optimization process. Default is True.
    seed : float, optional
        Random seed for reproducibility. Default is 0.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Affinity,
        kwargs_affinity_out: dict = {},
        n_components: int = 2,
        loss_fn: str = "square_loss",
        kwargs_loss: dict = {},
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        lr: float = 1e0,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        tol: float = 1e-3,
        max_iter: int = 1000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        early_exaggeration: float = None,
        early_exaggeration_iter: int = None,
        tolog: bool = False,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
        seed: float = 0,
    ):
        super().__init__(
            n_components=n_components, device=device, keops=keops, verbose=verbose
        )

        assert optimizer in OPTIMIZERS, f"Optimizer {optimizer} not supported."
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        assert loss_fn in LOSS_DICT, f"Loss function {loss_fn} not supported."
        self.loss_fn = loss_fn
        self.kwargs_loss = kwargs_loss

        self.init = init
        self.init_scaling = init_scaling

        self.tolog = tolog
        self.verbose = verbose
        self.seed = seed

        # --- check affinity_in ---
        if not isinstance(affinity_in, Affinity) and not affinity_in == "precomputed":
            raise ValueError(
                '[TorchDR] affinity_in must be an Affinity instance or "precomputed".'
            )
        self.affinity_in = affinity_in

        # --- check affinity_out ---
        if not isinstance(affinity_out, Affinity):
            raise ValueError("[TorchDR] affinity_out must be an Affinity instance.")
        self.affinity_out = affinity_out
        self.kwargs_affinity_out = kwargs_affinity_out

        if early_exaggeration is None or early_exaggeration_iter is None:
            self.early_exaggeration = 1
            early_exaggeration_iter = None
        else:
            self.early_exaggeration = early_exaggeration
            self.early_exaggeration_iter = early_exaggeration_iter

    @handle_backend
    def fit_transform(self, X: torch.Tensor | np.ndarray, y=None):
        """
        Fits the model to the provided data and returns the transformed data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data of shape (n_samples, n_features).
        y : None
            Ignored.

        Returns
        -------
        embedding_ : torch.Tensor
            The embedding of the input data.
        """
        self._fit(X)
        return self.embedding_

    def fit(self, X: torch.Tensor | np.ndarray, y=None):
        """
        Fits the model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data of shape (n_samples, n_features).
        y : None
            Ignored.

        Returns
        -------
        self : AffinityMatcher
            The fitted AffinityMatcher instance.
        """
        super().fit(X)
        self.fit_transform(X)
        return self

    def _fit(self, X: torch.Tensor):
        self.n_samples_in_, self.n_features_in_ = X.shape

        self._check_n_neighbors(self.n_samples_in_)

        # --- check if affinity_in is precomputed else compute it ---
        if self.affinity_in == "precomputed":
            if self.n_features_in_ != self.n_samples_in_:
                raise ValueError(
                    '[TorchDR] (Error) : When affinity_in="precomputed" the input X '
                    "in fit must be a tensor of lazy tensor of shape "
                    "(n_samples, n_samples)."
                )
            check_nonnegativity(X)
            self.PX_ = X
        else:
            self.PX_ = self.affinity_in.fit_transform(X)

        if hasattr(self, "early_exaggeration"):
            self.early_exaggeration_ = self.early_exaggeration

        self._init_embedding(X)
        self._set_params()
        optimizer = self._set_optimizer()
        scheduler = self._set_scheduler(optimizer)

        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for k in pbar:
            optimizer.zero_grad()
            loss = self._loss()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            check_NaNs(
                self.embedding_,
                msg="[TorchDR] AffinityMatcher : NaNs in the embeddings "
                f"at iter {k}.",
            )

            if self.verbose:
                pbar.set_description(f"Loss : {loss.item():.2e}")

            if (  # stop early exaggeration phase
                hasattr(self, "early_exaggeration")
                and k == self.early_exaggeration_iter
            ):
                self.early_exaggeration_ = 1
                optimizer = self._set_optimizer()
                self._set_scheduler(optimizer)

        self.n_iter_ = k

        return self

    def _loss(self):
        Q = self.affinity_out.fit_transform(self.embedding_, **self.kwargs_affinity_out)
        loss = LOSS_DICT[self.loss_fn](self.PX_, Q, **self.kwargs_loss)
        return loss

    def _set_params(self):
        self.params_ = [{"params": self.embedding_}]
        return self.params_

    def _set_optimizer(self):
        optimizer = OPTIMIZERS[self.optimizer](
            self.params_, lr=self.lr, **(self.optimizer_kwargs or {})
        )
        return optimizer

    def _set_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.scheduler == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1, total_iters=0
            )

        elif self.scheduler == "linear":
            linear_decay = lambda epoch: (1 - epoch / self.max_iter)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_decay)

        elif self.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        elif self.scheduler == "exponential":  # param gamma
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **(self.scheduler_kwargs or {})
            )

        else:
            raise ValueError(f"[TorchDR] scheduler : {self.scheduler} not supported.")

    def _init_embedding(self, X):
        n = X.shape[0]

        if self.init == "normal":
            generator = torch.Generator(device=X.device).manual_seed(self.seed)
            embedding_ = torch.randn(
                n,
                self.n_components,
                device=X.device,
                dtype=X.dtype,
                generator=generator,
            )

        elif self.init == "pca":
            embedding_ = PCA(n_components=self.n_components).fit_transform(X)

        else:
            raise ValueError(
                f"[TorchDR] init : {self.init} not supported in AffinityMatcher."
            )

        self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()
        return self.embedding_.requires_grad_()

    def _check_n_neighbors(self, n):
        param_list = ["perplexity", "n_neighbors"]

        for param_name in param_list:
            if hasattr(self, param_name):
                param_value = getattr(self, param_name)
                if n <= param_value:
                    if self.verbose:
                        print(
                            "[TorchDR] WARNING : Number of samples is smaller than "
                            f"{param_name} ({n} <= {param_value}), setting "
                            f"{param_name} to {n//2} (which corresponds to n//2)."
                        )
                    new_value = n // 2
                    setattr(self, param_name + "_", new_value)
                    setattr(self.affinity_in, param_name, new_value)

        return self


class BatchedAffinityMatcher(AffinityMatcher):
    r"""
    Performs dimensionality reduction by matching two batched affinity matrices.

    It amounts to solving the following optimization problem:

    .. math::

        \min_{\mathbf{Z}} \: \sum_{ij} L( [\mathbf{A_X}]_{ij}, [\mathbf{A_Z}]_{ij}) \:.

    Optimization of the embedding is perfomed using torch autodiff.

    Parameters
    ----------
    affinity_in : Affinity
        The affinity object for the input space.
    affinity_out : Affinity
        The affinity object for the output space.
    kwargs_affinity_out : dict, optional
        Additional keyword arguments for the affinity_out fit_transform method.
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    optimizer : str, optional
        Optimizer to use for the optimization. Default is "Adam".
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e0.
    scheduler : str, optional
        Learning rate scheduler. Default is "constant".
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    init : str, optional
        Initialization method for the embedding. Default is "pca".
    init_scaling : float, optional
        Scaling factor for the initial embedding. Default is 1e-4.
    early_exaggeration : int, optional
        Early exaggeration factor, by default None.
    early_exaggeration_iter : int, optional
        Number of iterations for early exaggeration, by default None.
    tolog : bool, optional
        If True, logs the optimization process. Default is False.
    device : str, optional
        Device to use for computations. Default is None.
    keops : bool, optional
        Whether to use KeOps for computations. Default is True.
    verbose : bool, optional
        Verbosity of the optimization process. Default is True.
    seed : float, optional
        Random seed for reproducibility. Default is 0.
    batch_size : int, optional
        Batch size for processing. Default is None.
    """  # noqa: E501

    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Affinity,
        kwargs_affinity_out: dict = {},
        n_components: int = 2,
        optimizer: str = "Adam",
        optimizer_kwargs: dict = None,
        lr: float = 1e0,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        tol: float = 1e-3,
        max_iter: int = 1000,
        init: str = "pca",
        init_scaling: float = 1e-4,
        early_exaggeration: float = None,
        early_exaggeration_iter: int = None,
        tolog: bool = False,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
        seed: float = 0,
        batch_size: int = None,
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
            early_exaggeration=early_exaggeration,
            early_exaggeration_iter=early_exaggeration_iter,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
            seed=seed,
        )

        self.batch_size = batch_size

    def batched_affinity_in_out(self, kwargs_affinity_out={}):
        """
        Returns batched affinity matrices for the input and output spaces.

        This method firsts generate permuted indices for batching. Using these indices, it computes the batched affinity matrices for both the input data and the embedded data.

        Parameters
        ----------
        kwargs_affinity_out : dict, optional
            Additional keyword arguments for the affinity_out fit_transform method.

        Returns
        -------
        batched_affinity_in_ : torch.Tensor or pykeops.torch.LazyTensor
            The batched affinity matrix for the input space.
        batched_affinity_out_ : torch.Tensor or pykeops.torch.LazyTensor
            The batched affinity matrix for the output space.
        """  # noqa: E501
        if (
            not hasattr(self, "batch_size_")
            or self.n_samples_in_ % getattr(self, "batch_size_") != 0
        ):
            self._set_batch_size()

        indices = self.generator_.permutation(self.n_samples_in_).reshape(
            -1, self.batch_size_
        )

        batched_affinity_in_ = self.affinity_in.get_batch(indices)
        batched_embedding_ = self.embedding_[indices]
        batched_affinity_out_ = self.affinity_out.fit_transform(
            batched_embedding_, **kwargs_affinity_out
        )

        return batched_affinity_in_, batched_affinity_out_

    def _instantiate_generator(self):
        self.generator_ = np.random.default_rng(
            seed=self.seed
        )  # we use numpy because torch.Generator is not picklable
        return self.generator_

    def _set_batch_size(self):
        if self.batch_size is not None and self.n_samples_in_ % self.batch_size == 0:
            self.batch_size_ = self.batch_size
            return self.batch_size_

        else:  # looking for a suitable batch_size
            for candidate_n_batches_ in np.arange(10, 1, -1):

                if self.n_samples_in_ % candidate_n_batches_ == 0:
                    self.batch_size_ = self.n_samples_in_ // candidate_n_batches_

                    if self.verbose:
                        print(
                            f"[TorchDR] WARNING : batch_size not provided or suitable, "
                            f"setting batch_size to {self.batch_size_} (for a total "
                            f"of {candidate_n_batches_} batches)."
                        )
                    return self.batch_size_

            raise ValueError(
                "[TorchDR] ERROR : could not find a suitable batch size. "
                "Please provide one. It should be a diviser of n_samples "
                f"({self.n_samples_in_} here). "
            )

    def _fit(self, X: torch.Tensor):
        self._instantiate_generator()
        super()._fit(X)
