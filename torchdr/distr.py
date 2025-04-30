"""Distributional Reduction module."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Dict, Optional, Type, Union

import numpy as np
import torch

from torchdr.affinity import Affinity
from torchdr.affinity_matcher import AffinityMatcher
from torchdr.utils import create_sparse_tensor_from_row_indices, to_torch


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
        init_OT_plan: Union[str, torch.Tensor, np.ndarray] = "random",
        n_iter_mirror_descent: int = 10,
        epsilon_mirror_descent: float = 1e-1,
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
        self.init_OT_plan = init_OT_plan
        self.n_iter_mirror_descent = n_iter_mirror_descent
        self.epsilon_mirror_descent = epsilon_mirror_descent

        if self.loss_fn == "square_loss":
            self.Loss = SquareLoss()
        elif self.loss_fn == "kl_loss":
            self.Loss = KLDivLoss()
        else:
            raise ValueError("[TorchDR] ERROR : loss_fn must be 'square_loss' or 'kl_loss'.")

        self._init_T()

    def _init_OT_plan(self):
        if isinstance(self.init_OT_plan, (torch.Tensor, np.ndarray)):
            self.OT_plan_ = to_torch(self.init_OT_plan, device=self.device)
            if self.OT_plan_.shape != (self.n_samples_in_, self.n_prototypes):
                raise ValueError(
                    f"[TorchDR] ERROR : init_OT_plan shape {self.OT_plan_.shape} "
                    "not compatible with (n_samples_in_, n_prototypes) = "
                    f"({self.n_samples_in_}, {self.n_prototypes})."
                )
        elif self.init_OT_plan == "random":
            OT_plan = torch.randn((self.n_samples_in_, self.n_prototypes), device=self.device)
            self.self.OT_plan_ = OT_plan / OT_plan.sum(-1, keepdim=True)
        else:
            raise ValueError("[TorchDR] ERROR : init_OT_plan must be 'random' or a torch.Tensor.")

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

    def _set_input_affinity(self, X: torch.Tensor):
        super()._set_input_affinity(X)
        # If sparsity is used, convert the affinity to a sparse tensor
        if hasattr(self, "NN_indices_"):
            self.PX_ = create_sparse_tensor_from_row_indices(
                self.PX_, self.NN_indices_, (self.n_samples_in_, self.n_samples_in_)
            )

    def _loss(self):
        one_N = torch.ones(self.n_samples_in_, device=self.device)

        Q = self.affinity_out(self.embedding_, **(self.kwargs_affinity_out or {}))
        Q_detached = Q.detach()  # Detach Q to prevent gradients flowing to the embeddings

        OT_plan = self.OT_plan_.clone()
        for _ in range(self.n_iter_mirror_descent):
            OT_plan.requires_grad_(True)

            q = OT_plan.sum(dim=0, keepdim=False)
            loss = self.Loss(self.PX_, Q_detached, one_N, q, OT_plan)
            loss.backward()

            # Mirror descent update
            with torch.no_grad():
                K = (OT_plan.grad - self.epsilon_mirror_descent * OT_plan).exp()
                OT_plan = K / K.sum(dim=1, keepdim=True)

        self.OT_plan_ = OT_plan
        q_converged = self.OT_plan_.sum(dim=0, keepdim=False)
        return self.Loss(self.PX_, Q, one_N, q_converged, self.OT_plan_)

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


class GromovWassersteinDecomposableLoss:
    """Base class for implementing decomposable loss functions for Gromov-Wasserstein problems.

    This class follows the decomposition framework for the Gromov-Wasserstein objective
    as described in Peyré et al. (2016): https://proceedings.mlr.press/v48/peyre16.pdf

    Subclasses must implement the decomposition functions f1, f2, h1, and h2.
    """

    def __call__(self, P, Q, p, q, OT_plan):
        one_p = torch.ones(p.shape[0], device=p.device)
        one_q = torch.ones(q.shape[0], device=q.device)
        L_kronecker_T = (
            self.f1(P) @ p @ one_q.T
            + one_p @ q.T @ self.f2(Q).T
            - self.h1(P) @ OT_plan @ self.h2(Q).T
        )
        return (L_kronecker_T * OT_plan).sum()

    def f1(x):
        raise NotImplementedError("[TorchDR] ERROR : f1 method is not implemented.")

    def f2(x):
        raise NotImplementedError("[TorchDR] ERROR : f2 method is not implemented.")

    def h1(x):
        raise NotImplementedError("[TorchDR] ERROR : h1 method is not implemented.")

    def h2(x):
        raise NotImplementedError("[TorchDR] ERROR : h2 method is not implemented.")


class SquareLoss(GromovWassersteinDecomposableLoss):
    def f1(X):
        return X**2

    def f2(X):
        return X**2

    def h1(X):
        return X

    def h2(X):
        return 2 * X


class KLDivLoss(GromovWassersteinDecomposableLoss):
    def f1(X):
        return X * X.log() - X

    def f2(X):
        return X

    def h1(X):
        return X

    def h2(X):
        return X.log()
