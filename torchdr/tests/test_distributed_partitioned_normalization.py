"""Regression test for distributed partitioned-loss normalization.

Attractive losses and negative-sampling repulsive losses are computed on
disjoint row chunks. Their unscaled sum must reconstruct the full loss before
gradient all-reduce. This complements the redundant-full repulsive-loss test,
whose terms instead require division by ``world_size``.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr import COSNE, SNE, TSNE, InfoTSNE, LargeVis


PARTITIONED_TERMS = [
    pytest.param(SNE, "_compute_attractive_loss", id="SNE-attractive"),
    pytest.param(TSNE, "_compute_attractive_loss", id="TSNE-attractive"),
    pytest.param(COSNE, "_compute_attractive_loss", id="COSNE-attractive"),
    pytest.param(InfoTSNE, "_compute_attractive_loss", id="InfoTSNE-attractive"),
    pytest.param(LargeVis, "_compute_attractive_loss", id="LargeVis-attractive"),
    pytest.param(InfoTSNE, "_compute_repulsive_loss", id="InfoTSNE-repulsive"),
    pytest.param(LargeVis, "_compute_repulsive_loss", id="LargeVis-repulsive"),
]


def _estimator_with_partitioned_state(cls, n_samples=24):
    generator = torch.Generator().manual_seed(0)
    estimator = cls(distributed=False)
    neighbors = torch.randint(0, n_samples, (n_samples, 5), generator=generator)
    negatives = torch.randint(0, n_samples, (n_samples, 6), generator=generator)
    affinity = torch.rand(n_samples, 5, generator=generator)
    # Global normalization makes disjoint row slices additive.
    affinity = affinity / affinity.sum()

    estimator.embedding_ = torch.randn(n_samples, 2, generator=generator) * 0.05
    estimator.n_samples_in_ = n_samples
    estimator.chunk_indices_ = torch.arange(n_samples)
    estimator.NN_indices_ = neighbors
    estimator.neg_indices_ = negatives
    estimator.affinity_in_ = affinity
    return estimator, neighbors, negatives, affinity


@pytest.mark.parametrize("cls,method", PARTITIONED_TERMS)
def test_partitioned_loss_sum_reconstructs_full(cls, method):
    estimator, neighbors, negatives, affinity = _estimator_with_partitioned_state(cls)
    loss = getattr(estimator, method)
    full = loss().sum()
    assert torch.isfinite(full) and full.abs() > 1e-6

    world_size = 3
    estimator.world_size = world_size
    total = full.new_zeros(())
    for rows in torch.chunk(torch.arange(estimator.n_samples_in_), world_size):
        estimator.chunk_indices_ = rows
        estimator.NN_indices_ = neighbors[rows]
        estimator.neg_indices_ = negatives[rows]
        estimator.affinity_in_ = affinity[rows]
        total += loss().sum()

    torch.testing.assert_close(total, full)
