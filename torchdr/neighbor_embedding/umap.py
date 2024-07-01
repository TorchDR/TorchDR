# -*- coding: utf-8 -*-
"""
UMAP algorithm
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.neighbor_embedding.base import NeighborEmbedding
from torchdr.affinity import (
    UMAPAffinityIn,
    UMAPAffinityOut,
)
from torchdr.utils import sum_all_axis_except_batch


class UMAP(NeighborEmbedding):
    def __init__(
        self,
        n_neighbors=30,
        n_components=2,
        min_dist=0.1,
        spread=1.0,
        a=None,
        b=None,
        lr=1.0,
        optimizer="Adam",
        optimizer_kwargs=None,
        scheduler: str = "constant",
        scheduler_kwargs: dict = None,
        init: str = "pca",
        init_scaling: float = 1e-4,
        tol: float = 1e-4,
        max_iter: int = 1000,
        tolog=False,
        device: str = None,
        keops: bool = False,
        verbose: bool = True,
        random_state: float = 0,
        coeff_attraction: float = 10.0,
        coeff_repulsion: float = 7.0,
        early_exaggeration_iter: int = 250,
        tol_affinity: float = 1e-3,
        max_iter_affinity: int = 100,
        metric_in: str = "euclidean",
        metric_out: str = "euclidean",
        batch_size: int | str = "auto",
    ):
        self.metric_in = metric_in
        self.metric_out = metric_out
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity
        self.n_neighbors = n_neighbors

        affinity_in = UMAPAffinityIn(
            n_neighbors=n_neighbors,
            metric=metric_in,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        affinity_out = UMAPAffinityOut(
            min_dist=min_dist,
            spread=spread,
            a=a,
            b=b,
            metric=metric_out,
            device=device,
            keops=keops,
            verbose=False,
        )

        super().__init__(
            affinity_in=affinity_in,
            affinity_out=affinity_out,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            tol=tol,
            max_iter=max_iter,
            lr=lr,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
            coeff_attraction=coeff_attraction,
            coeff_repulsion=coeff_repulsion,
            early_exaggeration_iter=early_exaggeration_iter,
            batch_size=batch_size,
        )

    @sum_all_axis_except_batch
    def _repulsive_loss(self, log_Q):
        Q = log_Q.exp()
        Q = Q / (Q + 1)  # trick from https://github.com/lmcinnes/umap/pull/856
        return -(1 - Q).log()
