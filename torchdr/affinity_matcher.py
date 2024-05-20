# -*- coding: utf-8 -*-
"""
Affinity matcher base classes
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#
# License: BSD 3-Clause License

import torch
<<<<<<< HEAD
import numpy as np
from typing import Union
from tqdm import tqdm
=======
from sklearn.base import TransformerMixin
>>>>>>> origin/main

from torchdr.utils import (
    OPTIMIZERS,
    check_nonnegativity,
    check_NaNs,
    handle_backend,
)
from torchdr.affinity import Affinity
from torchdr.spectral import PCA
from torchdr.base import DRModule


LOG_LOSSES = ["kl_loss", "cross_entropy_loss"]
LOSSES = LOG_LOSSES + ["square_loss"]


class AffinityMatcher(DRModule, TransformerMixin):
    def __init__(
        self,
        affinity_data: Affinity,
        affinity_embedding: Affinity,
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
        tolog: bool = False,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            n_components=n_components, device=device, keops=keops, verbose=verbose
        )

        assert optimizer in OPTIMIZERS, f"Optimizer {optimizer} not supported."
        self.optimizer = optimizer
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.init = init
        self.init_scaling = init_scaling

        self.tolog = tolog
        self.verbose = verbose

        # --- check affinity_data ---
        if (
            not isinstance(affinity_data, Affinity)
            and not affinity_data == "precomputed"
        ):
            raise ValueError(
                '[TorchDR] affinity_data must be an Affinity instance or "precomputed".'
            )
        self.affinity_data = affinity_data

        # --- check affinity_embedding ---
        if not isinstance(affinity_embedding, Affinity):
            raise ValueError(
                "[TorchDR] affinity_embedding must be an Affinity instance."
            )
        self.affinity_embedding = affinity_embedding

<<<<<<< HEAD
    def fit(self, X: Union[torch.Tensor, np.ndarray]):
=======
    def fit(self, X, y=None):
>>>>>>> origin/main
        super().fit(X)

        n = self.data_.shape[0]

        # --- check if affinity_data is precomputed else compute it ---
        if self.affinity_data == "precomputed":
            if self.data_.shape[1] != n:
                raise ValueError(
                    '[TorchDR] (Error) : When affinity_data="precomputed" the input X '
                    "in fit must be a tensor of lazy tensor of shape "
                    "(n_samples, n_samples)."
                )
            check_nonnegativity(self.data_)
            self.PX_ = self.data_
        else:
            self.PX_ = self.affinity_data.fit_transform(self.data_)

<<<<<<< HEAD
        self._init_embedding()
        self._set_params()
        optimizer = self._set_optimizer()
        scheduler = self._set_scheduler(optimizer)
=======
        # --- initialize embedding ---
        if self.init == "random":
            embedding_ = torch.randn(
                n, self.n_components, device=self.data_.device, dtype=self.data_.dtype
            )
        elif self.init == "pca":
            embedding_ = PCA(n_components=self.n_components).fit_transform(self.data_)
        else:
            raise ValueError(
                f"[TorchDR] {self.init} init not (yet) supported in AffinityMatcher."
            )
        embedding_ = embedding_ / embedding_[:, 0].std() * self.init_scaling

        embedding_.requires_grad = True
        optimizer_kwargs = self.optimizer_kwargs or {}
        optimizer = OPTIMIZERS[self.optimizer](
            [embedding_], lr=self.lr, **optimizer_kwargs
        )

        scheduler = self._make_scheduler(optimizer)
>>>>>>> origin/main

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
                and self.early_exaggeration != 1
            ):
                self.early_exaggeration = 1
                self._set_optimizer()
                self._set_scheduler(optimizer)

        return self

    @handle_backend
    def transform(self, X: Union[torch.Tensor, np.ndarray]):
        if not hasattr(self, "embedding_"):
            self.fit(X)
            assert hasattr(
                self, "embedding_"
            ), "The embedding embedding_ should be computed in fit method."
        return self.embedding_  # type: ignore

    def _set_params(self):
        self.params_ = [{"params": self.embedding_}]
        return self.params_

    def _set_optimizer(self):
        optimizer = OPTIMIZERS[self.optimizer](
            self.params_, lr=self.lr, **self.optimizer_kwargs
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
                optimizer, **self.scheduler_kwargs
            )

        else:
            raise ValueError(f"[TorchDR] scheduler : {self.scheduler} not supported.")

    def _init_embedding(self):
        n = self.data_.shape[0]

        if self.init == "normal":
            embedding_ = torch.randn(
                n, self.n_components, device=self.data_.device, dtype=self.data_.dtype
            )

        elif self.init == "pca":
            embedding_ = PCA(n_components=self.n_components).fit_transform(self.data_)

        else:
            raise ValueError(
                f"[TorchDR] init : {self.init} not supported in AffinityMatcher."
            )

        self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()
        return self.embedding_.requires_grad_()
