# -*- coding: utf-8 -*-
"""
Noise-constrastive SNE algorithms
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.affinity_matcher import BatchedAffinityMatcher
from torchdr.affinity import L2SymmetricEntropicAffinity, StudentAffinity
from torchdr.losses import cross_entropy_loss
from torchdr.utils import logsumexp_red


class InfoTSNE(BatchedAffinityMatcher):
    def __init__(
        self,
        perplexity=30,
        n_components=2,
        optimizer="Adam",
        optimizer_kwargs=None,
        lr=1.0,
        scheduler="constant",
        init="pca",
        init_scaling=1e-4,
        metric_in="euclidean",
        metric_out="euclidean",
        tol=1e-4,
        max_iter=1000,
        tol_affinity=1e-3,
        max_iter_affinity=100,
        tolog=False,
        device=None,
        keops=True,
        verbose=True,
        negative_samples=5,
        batch_size=None,
    ):

        self.metric_in = metric_in
        self.metric_out = metric_out
        self.perplexity = perplexity
        self.max_iter_affinity = max_iter_affinity
        self.tol_affinity = tol_affinity

        affinity_in = L2SymmetricEntropicAffinity(
            perplexity=perplexity,
            metric=metric_in,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        affinity_out = StudentAffinity(
            metric=metric_out,
            normalization_dim=None,  # we perform normalization when computing the loss
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
            init=init,
            init_scaling=init_scaling,
            tolog=tolog,
            device=device,
            keops=keops,
            verbose=verbose,
        )

        self.negative_samples = negative_samples
        self.batch_size = batch_size

    def _loss(self):
        """
        Dimensionality reduction objective.
        """
        PX_batch, Z_batch = self._batched_affinity_and_embedding()

        log_Q = self.affinity_out.fit_transform(Z_batch, log=True)
        info_log_Q = log_Q - logsumexp_red(log_Q, dim=1)

        losses = cross_entropy_loss(PX_batch, info_log_Q, log_Q=True)
        loss = losses.sum()

        return loss
