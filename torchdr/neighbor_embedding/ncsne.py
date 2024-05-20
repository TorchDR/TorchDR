# -*- coding: utf-8 -*-
"""
Noise-constrastive SNE algorithms
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.affinity_matcher import AffinityMatcher
from torchdr.affinity import L2SymmetricEntropicAffinity, StudentAffinity
import torch
from torchdr.utils import logsumexp_red, cross_entropy_loss


class InfoTSNE(AffinityMatcher):
    def __init__(
        self,
        perplexity,
        n_components=2,
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
        tolog=False,
        device=None,
        keops=True,
        verbose=True,
        negative_samples=5,
        batch_size=1000,
    ):

        entropic_affinity = L2SymmetricEntropicAffinity(
            perplexity=perplexity,
            metric=metric,
            tol=tol_affinity,
            max_iter=max_iter_affinity,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        affinity_embedding = StudentAffinity(
            metric=metric,
            dim_normalization=None,  # we perform normalization when computing the loss
            device=device,
            keops=keops,
            verbose=False,
        )

        super().__init__(
            affinity_data=entropic_affinity,
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

        self.negative_samples = negative_samples
        self.batch_size = batch_size

    def _loss(self):
        """
        Dimensionality reduction objective.
        """
        N = self.data_.shape[0]
        indices = torch.randperm(N)
        indices = indices.reshape(-1, self.batch_size)

        PX_batch = self.affinity_data.get_batch(indices)
        Z_batch = self.embedding_[indices]

        log_Q = self.affinity_embedding.fit_transform(Z_batch, log=True)
        info_log_Q = log_Q - logsumexp_red(log_Q, dim=1)

        losses = cross_entropy_loss(PX_batch, info_log_Q, log_Q=True)
        loss = losses.sum()

        return loss
