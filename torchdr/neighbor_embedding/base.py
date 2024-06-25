# -*- coding: utf-8 -*-
"""
Base classes for Neighbor Embedding methods
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.affinity_matcher import BatchedAffinityMatcher


class NeighborEmbedding(AffinityMatcher):

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
        coeff_attraction: float = 1.0,
        coeff_repulsion: float = 1.0,
        tolog: bool = False,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
        seed: float = 0,
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

        self.coeff_attraction = coeff_attraction
        self.coeff_repulsion = coeff_repulsion


class BatchedNeighborEmbedding(BatchedAffinityMatcher):

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
        batch_size: int = None,
        coeff_attraction: float = 1.0,
        coeff_repulsion: float = 1.0,
        tolog: bool = False,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
        seed: float = 0,
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
            batch_size=batch_size,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
            seed=seed,
        )

        self.coeff_attraction = coeff_attraction
        self.coeff_repulsion = coeff_repulsion
