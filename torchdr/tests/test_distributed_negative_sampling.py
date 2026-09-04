"""Regression tests for distributed negative-sampling exclusion (global coordinates).

In distributed neighbor embedding the output embedding is fully *replicated* on
every rank, while each rank owns only a contiguous chunk of query rows
``[chunk_start, chunk_start + chunk_size)`` (stored as ``chunk_indices_``). The
repulsive term is approximated by negative sampling, and for the sampled
negatives to be correct in this layout the exclusion setup in
:meth:`NegativeSamplingNeighborEmbedding.on_affinity_computation_end` must be
expressed in **global** coordinates:

* the per-row self-exclusion is the row's **global** index
  (``chunk_indices_ = arange(chunk_start, chunk_start + chunk_size)``), not its
  local position within the chunk; and
* negatives are drawn from the **full** candidate range ``n_samples_in_`` -- the
  whole replicated dataset -- not the local ``chunk_size``.

Two one-token regressions would silently corrupt this while leaving every
existing test green. Because the drawn indices stay in-range and the embedding
stays finite and bit-identical across ranks, the GPU smoke suite's
finiteness/replication assertions keep passing:

* **R1 -- local self-index:** using ``arange(chunk_size)`` instead of
  ``chunk_indices_`` self-excludes the wrong rows, so a point can be drawn as its
  own negative (repelling against itself) while another point is never excluded.
* **R2 -- chunk-local candidate range:** passing ``chunk_size`` instead of
  ``n_samples_in_`` restricts every rank to the first ``chunk_size`` global rows,
  so a rank owning rows ``[30, 70)`` of 100 never repels against ~60% of the
  dataset.

Both are invisible single-process, because there ``chunk_start == 0`` (global ==
local) and ``chunk_size == n_samples`` (the two ranges coincide). The tests below
place an estimator on a **non-zero-offset middle chunk** without a process group
and assert the exclusion buffers use global indices over the full pool. They
cover every distributed-capable negative-sampling estimator
(:class:`~torchdr.InfoTSNE`, :class:`~torchdr.LargeVis`, :class:`~torchdr.UMAP`);
:class:`~torchdr.PACMAP` is excluded by design -- it raises on ``distributed``.

The construction mirrors ``TestChunkStartOffset._model_with_chunk`` in
``test_distributed.py``: manual ``world_size`` / chunk bounds on CPU, no
NCCL/Gloo, no fit.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr import InfoTSNE, LargeVis, UMAP

# Every distributed-capable estimator whose negative sampling routes through
# ``NegativeSamplingNeighborEmbedding.on_affinity_computation_end``. UMAP reaches
# it via ``super()`` in its own override; InfoTSNE/LargeVis use it directly.
# PACMAP is absent on purpose: its constructor rejects ``distributed=True``.
DISTRIBUTED_NS_ESTIMATORS = [InfoTSNE, LargeVis, UMAP]
_IDS = [cls.__name__ for cls in DISTRIBUTED_NS_ESTIMATORS]

# A middle chunk of a 100-row dataset owned by a non-zero-offset rank. Using a
# *middle* chunk (not the tail) keeps ``chunk_start > 0`` and
# ``chunk_start + chunk_size < n_samples`` simultaneously, so R1 (global vs local
# index) and R2 (full vs chunk range) are both exercised crisply.
CHUNK_START, CHUNK_SIZE, N_SAMPLES = 30, 40, 100
K_NEIGHBORS = 4


def _sampler_on_middle_chunk(cls, exclude_neighbors=False):
    """Build a CPU estimator carrying just the distributed chunk state needed to
    run ``on_affinity_computation_end`` -- no process group, no fit.

    ``NN_indices_``/``affinity_in_`` are always provided: UMAP's override consumes
    them (and then deletes them) for edge flattening, and the exclude-neighbors
    path reads ``NN_indices_``. Neighbor indices are **global**, matching what a
    distributed ``SparseAffinity`` records for the chunk's rows.

    Returns the model and the tensor of this chunk's global row indices.
    """
    model = cls(
        n_components=2,
        exclude_neighbors_from_negative_sampling=exclude_neighbors,
    )
    model.rank = 0
    model.world_size = 2
    model.device_ = torch.device("cpu")
    model.n_samples_in_ = N_SAMPLES
    # Mirror what a distributed SparseAffinity records after a distributed call.
    model.affinity_in.chunk_start_ = CHUNK_START
    model.affinity_in.chunk_size_ = CHUNK_SIZE

    global_rows = torch.arange(CHUNK_START, CHUNK_START + CHUNK_SIZE)
    # K distinct global neighbor ids per row, none equal to the row itself
    # (offsets +1..+K never wrap back onto the row for this chunk).
    neighbors = (global_rows.unsqueeze(1) + 1 + torch.arange(K_NEIGHBORS)) % N_SAMPLES
    model.NN_indices_ = neighbors
    model.affinity_in_ = torch.rand(CHUNK_SIZE, K_NEIGHBORS) + 0.1

    model.on_affinity_computation_end()
    return model, global_rows


@pytest.mark.parametrize("cls", DISTRIBUTED_NS_ESTIMATORS, ids=_IDS)
def test_self_exclusion_uses_global_row_index(cls):
    """The per-row self-exclusion must be the global row index (kills R1)."""
    model, global_rows = _sampler_on_middle_chunk(cls, exclude_neighbors=False)

    # Exactly one exclusion per row (the self index), and it is the GLOBAL id.
    assert model.negative_exclusion_indices_.shape == (CHUNK_SIZE, 1)
    assert torch.equal(model.negative_exclusion_indices_[:, 0], global_rows), (
        f"{cls.__name__}: self-exclusion must be the global row index "
        f"arange({CHUNK_START}, {CHUNK_START + CHUNK_SIZE}); using the local "
        "chunk position excludes the wrong rows and lets a point be sampled as "
        "its own negative"
    )


@pytest.mark.parametrize("cls", DISTRIBUTED_NS_ESTIMATORS, ids=_IDS)
def test_negatives_span_the_full_replicated_dataset(cls):
    """Negatives are drawn from all ``n_samples_in_`` candidates (kills R2)."""
    model, global_rows = _sampler_on_middle_chunk(cls, exclude_neighbors=False)

    # Candidate pool is the full dataset minus the single self-exclusion. Sizing
    # it to ``chunk_size`` instead would give ``CHUNK_SIZE - 1`` here.
    assert torch.equal(
        model.negative_available_counts_,
        torch.full((CHUNK_SIZE,), N_SAMPLES - 1),
    ), (
        f"{cls.__name__}: each row must draw from n_samples_in_ candidates "
        "(minus itself); a chunk_size-sized pool restricts repulsion to the "
        "first chunk_size global rows"
    )

    # End-to-end: draws cover the whole global range, never equal the row's own
    # global index, and reach rows on BOTH sides of this chunk. Under R2 the pool
    # is [0, chunk_size), so no draw could reach rows >= chunk_start + chunk_size.
    torch.manual_seed(0)
    draws = model._draw_with_exclusions(
        model.negative_adjusted_exclusion_,
        model.negative_available_counts_,
        n_draws=2000,
    )
    assert draws.min() >= 0 and draws.max() < N_SAMPLES
    assert not (draws == global_rows.unsqueeze(1)).any(), (
        f"{cls.__name__}: a point was sampled as its own negative"
    )
    assert (draws >= CHUNK_START + CHUNK_SIZE).any(), (
        f"{cls.__name__}: negatives never reached rows above this chunk -- the "
        "candidate pool is not the full dataset"
    )
    assert (draws < CHUNK_START).any()


@pytest.mark.parametrize("cls", DISTRIBUTED_NS_ESTIMATORS, ids=_IDS)
def test_neighbor_exclusion_uses_global_indices(cls):
    """With neighbor exclusion on, both the global self index and the global
    neighbor indices are excluded from the full pool (kills R1 and R2)."""
    model, global_rows = _sampler_on_middle_chunk(cls, exclude_neighbors=True)
    neighbors = (global_rows.unsqueeze(1) + 1 + torch.arange(K_NEIGHBORS)) % N_SAMPLES
    excl = model.negative_exclusion_indices_

    assert (excl == global_rows.unsqueeze(1)).any(dim=1).all(), (
        f"{cls.__name__}: the global self index is missing from the exclusion "
        "set (a local self index was used)"
    )
    for j in range(K_NEIGHBORS):
        assert (excl == neighbors[:, j : j + 1]).any(dim=1).all(), (
            f"{cls.__name__}: a global neighbor index is missing from the exclusion set"
        )

    # self + K distinct neighbors excluded from the full n_samples_in_ pool.
    assert torch.equal(
        model.negative_available_counts_,
        torch.full((CHUNK_SIZE,), N_SAMPLES - (K_NEIGHBORS + 1)),
    ), f"{cls.__name__}: candidate pool must span n_samples_in_, not chunk_size"


@pytest.mark.parametrize("cls", DISTRIBUTED_NS_ESTIMATORS, ids=_IDS)
def test_neighbor_rows_must_match_chunk_size(cls):
    """Misaligned neighbor rows are rejected, not silently mis-broadcast."""
    model = cls(n_components=2, exclude_neighbors_from_negative_sampling=True)
    model.rank = 0
    model.world_size = 2
    model.device_ = torch.device("cpu")
    model.n_samples_in_ = N_SAMPLES
    model.affinity_in.chunk_start_ = CHUNK_START
    model.affinity_in.chunk_size_ = CHUNK_SIZE
    # NN_indices_ with the wrong number of rows must raise rather than align
    # neighbors to the wrong chunk rows.
    model.NN_indices_ = torch.zeros(CHUNK_SIZE + 1, K_NEIGHBORS, dtype=torch.long)
    model.affinity_in_ = torch.rand(CHUNK_SIZE + 1, K_NEIGHBORS) + 0.1

    with pytest.raises(ValueError, match="rows for chunk size"):
        model.on_affinity_computation_end()
