"""Tests for distributed repulsive-loss normalization.

SNE, TSNE, and COSNE recompute the full repulsive loss on every rank. Their
loss must therefore cancel the subsequent SUM all-reduce with ``world_size``.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr import COSNE, SNE, TSNE


REDUNDANT_REPULSION_ESTIMATORS = [SNE, TSNE, COSNE]


def _estimator_with_repulsive_state(estimator, n_samples=24):
    generator = torch.Generator().manual_seed(0)
    model = estimator(distributed=False)
    model.embedding_ = torch.randn(n_samples, 2, generator=generator) * 0.05
    model.n_samples_in_ = n_samples
    # COSNE additionally uses input norms in its repulsive term.
    model.X_norm = torch.randn(n_samples, 5, generator=generator).square().sum(-1)
    return model


@pytest.mark.parametrize("estimator", REDUNDANT_REPULSION_ESTIMATORS)
def test_redundant_repulsion_cancels_gradient_sum(estimator):
    model = _estimator_with_repulsive_state(estimator)

    model.world_size = 1
    reference = model._compute_repulsive_loss()
    assert torch.isfinite(reference) and reference.abs() > 1e-6

    model.world_size = 3
    distributed = model._compute_repulsive_loss()
    torch.testing.assert_close(distributed * model.world_size, reference)
