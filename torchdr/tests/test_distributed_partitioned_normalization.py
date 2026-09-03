"""Regression tests for the distributed *partitioned* loss normalization.

In distributed neighbor embedding the per-iteration gradient is summed across
ranks with ``dist.all_reduce(..., op=ReduceOp.SUM)`` (see
:meth:`torchdr.affinity_matcher.AffinityMatcher._training_step`). For that plain
``SUM`` to reconstruct the true full-data gradient, every loss term must fall
into exactly one of two conventions:

* **Redundant-full** (the repulsive loss of :class:`~torchdr.SNE`,
  :class:`~torchdr.TSNE`, :class:`~torchdr.COSNE`): every rank recomputes the
  *entire* term from the fully-replicated embedding, so a ``SUM`` all-reduce
  would multiply it by ``world_size``. These terms divide by ``world_size`` to
  cancel that factor. This half is locked by
  ``test_distributed_repulsive_normalization``.
* **Partitioned** (the attractive loss of *all* neighbor embeddings, and the
  negative-sampling repulsive loss of :class:`~torchdr.InfoTSNE` and
  :class:`~torchdr.LargeVis`): each rank evaluates the term only on its own row
  chunk (``query_indices=self.chunk_indices_``), so the disjoint per-rank sums
  already reconstruct the total and **no** ``/ world_size`` division is applied.

This file locks the *partitioned* half. The failure it guards against is the
symmetric mistake to the one guarded by the redundant-full tests: "helpfully"
adding a ``/ world_size`` factor to a partitioned term for consistency with the
redundant-full ones. Doing so under-scales attraction (and the sampled
repulsion) by ``world_size`` -- at ``world_size=4`` the attractive gradient
becomes a quarter of its correct value, so clusters never tighten. Crucially
this is invisible to the existing distributed tests: the broken term is still
finite and, because every rank divides identically, still bit-identical across
ranks, so the finiteness and replication assertions in the GPU suite keep
passing while the embedding is silently wrong.

Both checks run on CPU without a process group. A partitioned term does not read
``world_size`` at all, so re-evaluating it at two world sizes on identical state
isolates the (absence of) normalization exactly; summing it over a disjoint row
partition -- each chunk evaluated at ``world_size = n_chunks`` -- mirrors the
real ``SUM`` all-reduce and must reproduce the single-process value.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr import COSNE, SNE, TSNE, InfoTSNE, LargeVis

# Each entry is (id, estimator, loss-method name). Every listed term is
# partitioned across ranks and must therefore be independent of ``world_size``.
PARTITIONED_TERMS = [
    ("SNE.attractive", SNE, "_compute_attractive_loss"),
    ("TSNE.attractive", TSNE, "_compute_attractive_loss"),
    ("COSNE.attractive", COSNE, "_compute_attractive_loss"),
    ("InfoTSNE.attractive", InfoTSNE, "_compute_attractive_loss"),
    ("LargeVis.attractive", LargeVis, "_compute_attractive_loss"),
    ("InfoTSNE.repulsive", InfoTSNE, "_compute_repulsive_loss"),
    ("LargeVis.repulsive", LargeVis, "_compute_repulsive_loss"),
]
_IDS = [term[0] for term in PARTITIONED_TERMS]


def _estimator_with_partitioned_state(
    cls, n_samples=24, n_components=2, n_neighbors=5, n_negatives=6
):
    """Build a non-distributed estimator carrying just enough fitted state to
    call the partitioned loss methods directly on CPU, without a process group.

    The partitioned losses read ``embedding_``, ``chunk_indices_`` (the rank's
    query rows), ``NN_indices_``/``affinity_in_`` (attractive) and
    ``neg_indices_``/``n_samples_in_`` (InfoTSNE/LargeVis repulsive). The full
    row set is used as the default chunk; the reconstruction test slices these
    tensors per chunk. The embedding is kept small so CoSNE's ``sqhyperbolic``
    metric stays inside the Poincare ball. ``affinity_in_`` is globally
    normalized (not per row) so that slicing rows partitions the loss additively.
    """
    torch.manual_seed(0)
    est = cls(distributed=False)
    all_rows = torch.arange(n_samples)
    nn_indices = torch.randint(0, n_samples, (n_samples, n_neighbors))
    neg_indices = torch.randint(0, n_samples, (n_samples, n_negatives))
    affinity = torch.rand(n_samples, n_neighbors)
    affinity = affinity / affinity.sum()

    est.embedding_ = torch.randn(n_samples, n_components) * 0.05
    est.n_samples_in_ = n_samples
    est.chunk_indices_ = all_rows
    est.NN_indices_ = nn_indices
    est.neg_indices_ = neg_indices
    est.affinity_in_ = affinity

    full_state = {
        "n_samples": n_samples,
        "NN_indices": nn_indices,
        "neg_indices": neg_indices,
        "affinity": affinity,
    }
    return est, full_state


@pytest.mark.parametrize(
    "cls,method", [(c, m) for _, c, m in PARTITIONED_TERMS], ids=_IDS
)
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_partitioned_loss_is_world_size_independent(cls, method, world_size):
    """A partitioned term must not change when ``world_size`` changes.

    Mirror of the redundant-full check: a spurious ``/ world_size`` division
    would make the loss scale as ``1 / world_size`` instead of staying constant.
    """
    est, _ = _estimator_with_partitioned_state(cls)

    est.world_size = 1
    base = getattr(est, method)()
    assert torch.isfinite(base), f"{method} on {cls.__name__} is not finite"
    # A non-degenerate baseline is required for this to be meaningful: a ~0 loss
    # would be trivially close to ``loss / world_size`` and hide the regression.
    assert base.abs() > 1e-6, f"{method} on {cls.__name__} baseline is degenerate"

    est.world_size = world_size
    scaled = getattr(est, method)()
    assert torch.allclose(scaled, base, rtol=1e-5, atol=1e-7), (
        f"{method} on {cls.__name__} changed with world_size={world_size} "
        f"(partitioned terms must not divide by world_size): "
        f"got {scaled.item()}, expected {base.item()}"
    )


@pytest.mark.parametrize(
    "cls,method", [(c, m) for _, c, m in PARTITIONED_TERMS], ids=_IDS
)
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_partitioned_loss_sum_reconstructs_full(cls, method, world_size):
    """Summing a partitioned term over a disjoint row partition -- each chunk
    evaluated at ``world_size = n_chunks`` -- must reproduce the single-process
    value.

    This is exactly the invariant the ``SUM`` all-reduce relies on. If the term
    were divided by ``world_size``, the ``world_size`` chunks would each be
    scaled by ``1 / world_size`` and their sum would be ``full / world_size``,
    not ``full``.
    """
    est, state = _estimator_with_partitioned_state(cls)
    n_samples = state["n_samples"]

    est.world_size = 1
    full = getattr(est, method)()
    assert torch.isfinite(full), f"{method} on {cls.__name__} is not finite"
    assert full.abs() > 1e-6, f"{method} on {cls.__name__} baseline is degenerate"

    est.world_size = world_size
    total = torch.zeros((), dtype=full.dtype)
    for rows in torch.chunk(torch.arange(n_samples), world_size):
        est.chunk_indices_ = rows
        est.NN_indices_ = state["NN_indices"][rows]
        est.neg_indices_ = state["neg_indices"][rows]
        est.affinity_in_ = state["affinity"][rows]
        total = total + getattr(est, method)()

    assert torch.allclose(total, full, rtol=1e-5, atol=1e-6), (
        f"{method} on {cls.__name__} did not reconstruct the full loss from "
        f"{world_size} chunks (partitioned terms must not divide by "
        f"world_size): got {total.item()}, expected {full.item()}"
    )
