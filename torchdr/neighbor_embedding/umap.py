# -*- coding: utf-8 -*-
"""
UMAP algorithm
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.affinity_matcher import AffinityMatcher
from torchdr.affinity import (
    UMAPAffinityData,
    UMAPAffinityEmbedding,
)
from torchdr.utils import binary_cross_entropy_loss


class UMAP(AffinityMatcher):
    def __init__(
        self,
        n_neighbors,
        n_components=2,
        min_dist=0.1,
        spread=1.0,
        a=None,
        b=None,
        optimizer="Adam",
        optimizer_kwargs=None,
        lr=1.0,
        scheduler="constant",
        init="pca",
        init_scaling=1e-4,
        metric="euclidean",
        tol=1e-4,
        max_iter=1000,
        tol_affinity=1e-3,
        max_iter_affinity=100,
        verbose=True,
        tolog=False,
        device=None,
        keops=True,
        coeff_repulsion=1.0,
    ):
        affinity_data = UMAPAffinityData(
            n_neighbors=n_neighbors,
            metric=metric,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        affinity_embedding = UMAPAffinityEmbedding(
            min_dist=min_dist,
            spread=spread,
            a=a,
            b=b,
            metric=metric,
            device=device,
            keops=keops,
            verbose=False,
        )

        super().__init__(
            affinity_data=affinity_data,
            affinity_embedding=affinity_embedding,
            n_components=n_components,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            tol=tol,
            max_iter=max_iter,
            lr=lr,
            scheduler=scheduler,
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
        )

        self.coeff_repulsion = coeff_repulsion

    def _loss(self, embedding_):
        """
        Dimensionality reduction objective.
        """

        Q = self.affinity_embedding.fit_transform(embedding_)
        Q = Q / (Q + 1)
        # Q = Q.clamp(1e-4, 1)
        return binary_cross_entropy_loss(
            self.PX_, Q, coeff_repulsion=self.coeff_repulsion
        )
