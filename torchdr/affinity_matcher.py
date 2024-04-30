# -*- coding: utf-8 -*-
"""
Affinity matcher base classes
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Titouan Vayer <titouan.vayer@inria.fr>
#
# License: BSD 3-Clause License

from tqdm import tqdm
import torch

from torchdr.utils import (
    OPTIMIZERS,
    cross_entropy_loss,
    square_loss,
    check_nonnegativity,
    check_NaNs,
    handle_backend,
)
from torchdr.affinity import Affinity, LogAffinity
from torchdr.spectral import PCA
from torchdr.base import DRModule

LOG_LOSSES = ["kl_loss", "cross_entropy_loss"]
LOSSES = LOG_LOSSES + ["square_loss"]


class AffinityMatcher(DRModule):
    def __init__(
        self,
        affinity_data,
        affinity_embedding,
        n_components,
        loss_fun="cross_entropy_loss",
        optimizer="Adam",
        optimizer_kwargs=None,
        lr=1e-3,
        scheduler="linear",
        tol=1e-6,
        max_iter=1000,
        init="pca",
        init_scaling=1e-4,
        tolog=False,
        device=None,
        keops=True,
        verbose=True,
    ):
        super().__init__(
            n_components=n_components, device=device, keops=keops, verbose=verbose
        )
        self.loss_fun = loss_fun

        assert optimizer in OPTIMIZERS, f"Optimizer {optimizer} not supported."
        self.optimizer = optimizer
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler = scheduler

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

    def fit(self, X):
        super().fit(X)

        n = self.X_.shape[0]

        # --- check if affinity_data is precomputed else compute it ---
        if self.affinity_data == "precomputed":
            if self.X_.shape[1] != n:
                raise ValueError(
                    '[TorchDR] (Error) : When affinity_data="precomputed" the input X '
                    "in fit must be a tensor of lazy tensor of shape "
                    "(n_samples, n_samples)."
                )
            check_nonnegativity(self.X_)
            self.PX_ = self.X_
        else:
            self.PX_ = self.affinity_data.fit_transform(self.X_)

        # --- initialize embedding ---
        if self.init == "random":
            Z_ = torch.randn(n, self.n_components).to(self.X_.device, self.X_.dtype)
        elif self.init == "pca":
            Z_ = PCA(n_components=self.n_components).fit_transform(self.X_)
        else:
            raise ValueError(
                f"[TorchDR] {self.init} init not (yet) supported in AffinityMatcher."
            )
        Z_ = Z_ / torch.std(Z_[:, 0]) * self.init_scaling

        Z_.requires_grad = True
        optimizer = OPTIMIZERS[self.optimizer](
            [Z_], lr=self.lr, **self.optimizer_kwargs
        )

        scheduler = self._make_scheduler(optimizer)

        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for k in pbar:
            optimizer.zero_grad()
            loss = self._loss(Z_)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            check_NaNs(Z_, msg=f"NaNs in the embeddings at iter {k}.")

            if self.verbose:
                pbar.set_description(f"Loss : {loss.item():.2e}")

        self.Z_ = Z_.detach()

    @handle_backend
    def transform(self, X):
        if not hasattr(self, "Z_"):
            self.fit(X)
            assert hasattr(
                self, "Z_"
            ), "The embedding Z_ should be computed in fit method."
        return self.Z_  # type: ignore

    def _make_scheduler(self, optimizer):
        if self.scheduler == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1, total_iters=0
            )

        elif self.scheduler == "linear":
            lambda1 = lambda epoch: (1 - epoch / self.max_iter)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        elif self.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        elif self.scheduler == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        else:
            raise ValueError(
                f"[TorchDR] {self.scheduler} scheduler not (yet) supported."
            )

    def _loss(self, Z_):
        """
        Dimensionality reduction objective.
        """
        if self.loss_fun == "cross_entropy_loss":
            if isinstance(self.affinity_embedding, LogAffinity):
                log_QZ_ = self.affinity_embedding.fit_transform(Z_, log=True)
                loss = cross_entropy_loss(self.PX_, log_QZ_, log_Q=True)
            else:
                QZ_ = self.affinity_embedding.fit_transform(Z_)
                loss = cross_entropy_loss(self.PX_, QZ_, log_Q=False)

        elif self.loss_fun == "square_loss":
            QZ_ = self.affinity_embedding.fit_transform(Z_)
            loss = square_loss(self.PX_, QZ_)

        else:
            raise ValueError(
                f"[TorchDR] {self.loss_fun} loss_fun not supported "
                "in Affinity Matcher."
            )

        return loss
