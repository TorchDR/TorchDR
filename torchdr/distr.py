"""Distributional Reduction module."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Type, Union

import numpy as np
import torch

from torchdr.affinity import Affinity
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.utils import to_torch


class DistR(AffinityMatcher):
    def __init__(
        self,
        affinity_in: Affinity,
        affinity_out: Affinity,
        kwargs_affinity_out: dict = {},
        n_components: int = 2,
        loss_fn: str = "square_loss",
        kwargs_loss: dict = {},
        optimizer: Union[str, Type[torch.optim.Optimizer]] = "Adam",
        optimizer_kwargs: Optional[Dict] = None,
        lr: float | str = 1e0,
        scheduler: Optional[Union[str, Type[torch.optim.lr_scheduler.LRScheduler]]] = None,
        scheduler_kwargs: Optional[Dict] = None,
        min_grad_norm: float = 1e-7,
        max_iter: int = 1000,
        init: str | torch.Tensor | np.ndarray = "pca",
        init_scaling: float = 1e-4,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        random_state: float = None,
        n_iter_check: int = 50,
        n_prototypes: int = 10,
    ):
        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            kwargs_affinity_out=kwargs_affinity_out,
            n_components=n_components,
            loss_fn=loss_fn,
            kwargs_loss=kwargs_loss,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            init=init,
            init_scaling=init_scaling,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            n_iter_check=n_iter_check,
        )
        self.n_prototypes = n_prototypes

    def _check_affinities(self, affinity_in, affinity_out, kwargs_affinity_out):
        # --- check affinity_out ---
        if not isinstance(affinity_out, Affinity):
            raise ValueError("[TorchDR] ERROR : affinity_out must be an Affinity instance.")
        self.affinity_out = affinity_out
        self.kwargs_affinity_out = kwargs_affinity_out

        # --- check affinity_in ---
        if not isinstance(affinity_in, Affinity) and not affinity_in == "precomputed":
            raise ValueError('[TorchDR] affinity_in must be an Affinity instance or "precomputed".')
        self.affinity_in = affinity_in

    def _loss(self):
        return super()._loss()

    def _init_embedding(self, X):
        if isinstance(self.init, (torch.Tensor, np.ndarray)):
            embedding_ = to_torch(self.init, device=self.device)
            if embedding_.shape != (self.n_prototypes, self.n_components):
                raise ValueError(
                    f"[TorchDR] ERROR : init shape {embedding_.shape} not compatible "
                    f"with (n, n_components) = ({self.n_prototypes}, {self.n_components})."
                )

        elif self.init == "normal" or self.init == "random":
            embedding_ = torch.randn(
                (self.n_prototypes, self.n_components),
                device=X.device if self.device == "auto" else self.device,
                dtype=X.dtype,
            )

        else:
            raise ValueError(
                f"[TorchDR] ERROR : init {self.init} not supported in {self.__class__.__name__}."
            )

        self.embedding_ = self.init_scaling * embedding_ / embedding_[:, 0].std()
        return self.embedding_.requires_grad_()
