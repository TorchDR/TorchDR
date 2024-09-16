# -*- coding: utf-8 -*-
"""Affinity matcher base classes."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from tqdm import tqdm
import warnings

from torchdr.utils import (
    OPTIMIZERS,
    check_nonnegativity,
    check_NaNs,
    handle_backend,
    to_torch,
)
from torchdr.affinity import (
    Affinity,
    LogAffinity,
    SparseLogAffinity,
    UnnormalizedAffinity,
)
from torchdr.spectral import PCA
from torchdr.base import DRModule
from torchdr.utils import square_loss, cross_entropy_loss

LOSS_DICT = {
    "square_loss": square_loss,
    "cross_entropy_loss": cross_entropy_loss,
}


class AffinityMatcher(DRModule):
    r"""Perform dimensionality reduction by matching two affinity matrices.

    It amounts to solving a problem of the form:

    .. math::

        \min_{\mathbf{Z}} \: \mathcal{L}( \mathbf{A_X}, \mathbf{A_Z})

    where :math:`\mathcal{L}` is a loss function, :math:`\mathbf{A_X}` is the
    input affinity matrix and :math:`\mathbf{A_Z}` is the affinity matrix of the
    embedding.

    The embedding optimization is performed using a first-order optimization method, with gradients calculated via PyTorch's automatic differentiation.

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
    optimizer : str, optional
        Optimizer to use for the optimization. Default is "Adam".
    optimizer_kwargs : dict, optional
        Additional keyword arguments for the optimizer.
    lr : float or 'auto', optional
        Learning rate for the optimizer. Default is 1e0.
    scheduler : str, optional
        Learning rate scheduler. Default is "constant".
    scheduler_kwargs : dict, optional
        Additional keyword arguments for the scheduler.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    init : str | torch.Tensor | np.ndarray, optional
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
        lr: float | str = 1e0,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        tol: float = 1e-7,
        max_iter: int = 1000,
        init: str | torch.Tensor | np.ndarray = "pca",
        init_scaling: float = 1e-4,
        tolog: bool = False,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
        random_state: float = 0,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
        )

        if optimizer not in OPTIMIZERS and optimizer != "auto":
            raise ValueError(f"[TorchDR] ERROR : Optimizer {optimizer} not supported.")

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        if loss_fn not in LOSS_DICT:
            raise ValueError(
                f"[TorchDR] ERROR : Loss function {loss_fn} not supported."
            )
        self.loss_fn = loss_fn
        self.kwargs_loss = kwargs_loss

        self.init = init
        self.init_scaling = init_scaling

        self.tolog = tolog
        self.verbose = verbose

        # --- check affinity_out ---
        if not isinstance(affinity_out, Affinity):
            raise ValueError(
                "[TorchDR] ERROR : affinity_out must be an Affinity instance."
            )
        self.affinity_out = affinity_out
        self.kwargs_affinity_out = kwargs_affinity_out

        # --- check affinity_in ---
        if not isinstance(affinity_in, Affinity) and not affinity_in == "precomputed":
            raise ValueError(
                '[TorchDR] affinity_in must be an Affinity instance or "precomputed".'
            )
        if getattr(affinity_in, "sparsity", False) and not isinstance(
            self.affinity_out, UnnormalizedAffinity
        ):
            warnings.warn(
                "[TorchDR] WARNING : affinity_out must be a UnnormalizedAffinity "
                "when affinity_in is sparse. Setting sparsity = False in affinity_in."
            )
            affinity_in._sparsity = False  # turn off sparsity
        self.affinity_in = affinity_in

    @handle_backend
    def fit_transform(self, X: torch.Tensor | np.ndarray, y=None):
        """Fit the model to the provided data and returns the transformed data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data.
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
        """Fit the model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data.
        y : None
            Ignored.

        Returns
        -------
        self : AffinityMatcher
            The fitted AffinityMatcher instance.
        """
        self.fit_transform(X)
        return self

    def _fit(self, X: torch.Tensor):
        self._instantiate_generator()

        self.n_samples_in_, self.n_features_in_ = X.shape

        # --- check if affinity_in is precomputed else compute it ---
        if self.affinity_in == "precomputed":
            if self.n_features_in_ != self.n_samples_in_:
                raise ValueError(
                    '[TorchDR] ERROR : When affinity_in="precomputed" the input X '
                    "in fit must be a tensor of lazy tensor of shape "
                    "(n_samples, n_samples)."
                )
            check_nonnegativity(X)
            self.PX_ = X
        else:
            if isinstance(self.affinity_in, SparseLogAffinity):
                self.PX_, self.NN_indices_ = self.affinity_in(X, return_indices=True)
            else:
                self.PX_ = self.affinity_in(X)

        self._init_embedding(X)
        self._set_params()
        self._set_learning_rate()
        self._set_optimizer()
        self._set_scheduler()

        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for k in pbar:
            self.optimizer_.zero_grad()
            loss = self._loss()
            loss.backward()

            grad_norm = self.embedding_.grad.norm(2).item()
            if grad_norm < self.tol:
                if self.verbose:
                    pbar.set_description(
                        f"[TorchDR] Convergence reached at iter {k} with grad norm: "
                        f"{grad_norm:.2e}."
                    )
                break

            self.optimizer_.step()
            self.scheduler_.step()

            check_NaNs(
                self.embedding_,
                msg="[TorchDR] ERROR AffinityMatcher : NaNs in the embeddings "
                f"at iter {k}.",
            )

            if self.verbose:
                pbar.set_description(
                    f"[TorchDR] DR Loss : {loss.item():.2e} | "
                    f"Grad norm : {grad_norm:.2e} "
                )

            self._additional_updates(k)

        self.n_iter_ = k

        return self

    def _loss(self):
        if (self.loss_fn == "cross_entropy_loss") and isinstance(
            self.affinity_out, LogAffinity
        ):
            self.kwargs_affinity_out.setdefault("log", True)
            self.kwargs_loss.setdefault("log", True)

        if getattr(self, "NN_indices_", None) is not None:
            Q = self.affinity_out(
                self.embedding_, indices=self.NN_indices_, **self.kwargs_affinity_out
            )
        else:
            Q = self.affinity_out(self.embedding_, **self.kwargs_affinity_out)
        loss = LOSS_DICT[self.loss_fn](self.PX_, Q, **self.kwargs_loss)
        return loss

    def _additional_updates(self, step):
        pass

    def _set_params(self):
        self.params_ = [{"params": self.embedding_}]
        return self.params_

    def _set_optimizer(self):
        self.optimizer_ = OPTIMIZERS[self.optimizer](
            self.params_, lr=self.lr_, **(self.optimizer_kwargs or {})
        )
        return self.optimizer_

    def _set_learning_rate(self):
        if self.lr == "auto":
            if self.verbose:
                warnings.warn(
                    "[TorchDR] WARNING : lr set to 'auto' without "
                    "any implemented rule. Setting lr=1.0 by default."
                )
            self.lr_ = 1.0
        else:
            self.lr_ = self.lr

    def _set_scheduler(self, n_iter=None):
        n_iter = n_iter or self.max_iter

        if not hasattr(self, "optimizer_"):
            raise ValueError(
                "[TorchDR] ERROR : optimizer not set. "
                "Please call _set_optimizer before _set_scheduler."
            )

        if self.scheduler == "constant":
            self.scheduler_ = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer_, factor=1, total_iters=0
            )

        elif self.scheduler == "linear":
            linear_decay = lambda epoch: (1 - epoch / n_iter)
            self.scheduler_ = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_, lr_lambda=linear_decay
            )

        elif self.scheduler == "cosine":
            self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_, T_max=n_iter
            )

        else:
            raise ValueError(
                f"[TorchDR] ERROR : scheduler {self.scheduler} not supported."
            )

        return self.scheduler_

    def _instantiate_generator(self):
        self.generator_ = np.random.default_rng(
            seed=self.random_state
        )  # we use numpy because torch.Generator is not picklable
        return self.generator_

    def _init_embedding(self, X):
        n = X.shape[0]

        if isinstance(self.init, (torch.Tensor, np.ndarray)):
            embedding_ = to_torch(self.init, device=self.device)

        elif self.init == "normal" or self.init == "random":
            embedding_ = torch.tensor(
                self.generator_.standard_normal(size=(n, self.n_components)),
                device=X.device if self.device == "auto" else self.device,
                dtype=X.dtype,
            )

        elif self.init == "pca":
            embedding_ = PCA(
                n_components=self.n_components, device=self.device
            ).fit_transform(X)

        else:
            raise ValueError(
                f"[TorchDR] ERROR : init {self.init} not supported in "
                f"{self.__class__.__name__}."
            )

        self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()
        return self.embedding_.requires_grad_()
