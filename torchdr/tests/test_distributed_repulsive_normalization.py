"""Regression tests for the distributed repulsive-loss normalization.

In distributed neighbor embedding the per-iteration gradient is summed across
ranks with ``dist.all_reduce(..., op=ReduceOp.SUM)`` (see
:meth:`torchdr.affinity_matcher.AffinityMatcher._training_step`). Two different
conventions keep that sum numerically correct for the repulsive term:

* **Redundant-full** (:class:`~torchdr.SNE`, :class:`~torchdr.TSNE`,
  :class:`~torchdr.COSNE`): every rank recomputes the *entire* repulsive loss
  from the fully-replicated embedding, so a plain ``SUM`` all-reduce would
  multiply the repulsive gradient by ``world_size``. Each of these estimators
  divides ``_compute_repulsive_loss`` by ``world_size`` to cancel that factor.
* **Partitioned** (:class:`~torchdr.InfoTSNE`, :class:`~torchdr.LargeVis`):
  each rank computes only its own chunk's contribution, so the disjoint sums
  already reconstruct the total and no division is applied.

This asymmetry is easy to get wrong and, crucially, invisible to the existing
distributed tests. Dropping the ``/ world_size`` factor leaves *every* rank with
the same over-scaled repulsive gradient, so the embedding stays replicated and
finite: the replication and finiteness assertions in the GPU suite still pass
while the embedding is silently over-repelled by a factor of ``world_size``.

These tests lock the factor directly, on CPU, without a process group: the
redundant-full repulsive loss depends on ``world_size`` only through the final
division, so re-evaluating it at two world sizes on identical state isolates the
normalization exactly.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr import COSNE, SNE, TSNE

# Estimators whose repulsive loss is recomputed in full on every rank and is
# therefore divided by ``world_size`` to cancel the SUM all-reduce.
REDUNDANT_FULL_ESTIMATORS = [SNE, TSNE, COSNE]


def _estimator_with_repulsive_state(cls, n_samples=24, n_components=2):
    """Build a non-distributed estimator carrying just enough fitted state to
    call ``_compute_repulsive_loss`` directly on CPU, without a process group.

    The redundant-full repulsive losses read only ``embedding_``, ``backend``,
    ``n_samples_in_`` and ``world_size`` (CoSNE additionally reads ``X_norm``,
    ``gamma`` and ``learning_rate_for_h_loss``, the latter two set at
    construction). A small embedding keeps CoSNE's points inside the Poincare
    ball, which the ``sqhyperbolic`` metric requires.
    """
    torch.manual_seed(0)
    est = cls(distributed=False)
    est.embedding_ = torch.randn(n_samples, n_components) * 0.05
    est.n_samples_in_ = n_samples
    # Only consumed by CoSNE's distance-preservation term; harmless otherwise.
    est.X_norm = (torch.randn(n_samples, 5) ** 2).sum(-1)
    return est


@pytest.mark.parametrize("cls", REDUNDANT_FULL_ESTIMATORS)
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_redundant_full_repulsive_loss_scaled_by_world_size(cls, world_size):
    """The redundant-full repulsive loss must scale as ``1 / world_size``.

    This is the factor that cancels the ``ReduceOp.SUM`` gradient all-reduce.
    """
    est = _estimator_with_repulsive_state(cls)

    est.world_size = 1
    base = est._compute_repulsive_loss()
    assert torch.isfinite(base), f"{cls.__name__} repulsive loss is not finite"
    # A non-degenerate baseline is required for the ratio below to be meaningful:
    # if ``base`` were ~0, ``base`` and ``base / world_size`` would be trivially
    # close and dropping the normalization would go undetected.
    assert base.abs() > 1e-6, f"{cls.__name__} repulsive baseline is degenerate"

    est.world_size = world_size
    scaled = est._compute_repulsive_loss()
    assert torch.allclose(scaled, base / world_size, rtol=1e-5, atol=1e-7), (
        f"{cls.__name__} repulsive loss did not scale by 1/{world_size}: "
        f"got {scaled.item()}, expected {(base / world_size).item()}"
    )
