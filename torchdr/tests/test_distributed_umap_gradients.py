"""Regression test for UMAP's distributed closed-form gradients.

In distributed mode the embedding is replicated on every rank while each rank
owns a contiguous row chunk ``[chunk_start, chunk_end)``. UMAP's closed-form
attractive and repulsive gradients therefore map their *local* edge sources
back to *global* embedding rows through ``chunk_indices_`` and index the
replicated embedding with the *global* ``attractive_target_`` / ``neg_indices_``
columns. A rank's gradient for its chunk must equal the corresponding slice of
the single-process, full-data gradient.

``test_umap_csr.py`` pins the flat (CSR) attractive gradient against a
brute-force reference, but only with ``chunk_indices_ = arange(n)`` -- a single
rank whose local rows already equal the global rows, where the local->global
mapping is the identity and so cannot expose a wrong-row bug. This test instead
exercises non-zero chunk starts (``world_size > 1``) for *both* the attractive
and repulsive gradients, so it fails if either seam is regressed to a local
index, e.g. ``self.embedding_[source]`` instead of
``self.embedding_[self.chunk_indices_[source]]`` in the attractive term, or a
local ``arange`` query in the repulsive term.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr.neighbor_embedding import UMAP


def _make_umap_state(n_samples=24, n_components=2, max_degree=4, n_negatives=6):
    """Deterministic replicated embedding and padded neighbor/negative grids."""
    generator = torch.Generator().manual_seed(0)
    embedding = torch.randn(
        n_samples, n_components, dtype=torch.float64, generator=generator
    )

    # Padded (n, max_degree) neighbor grid with -1 padding and *global* columns.
    # Variable per-row degree (2..4) exercises the segmented reduction.
    nn_grid = torch.full((n_samples, max_degree), -1, dtype=torch.long)
    for i in range(n_samples):
        degree = 2 + (i % (max_degree - 1))
        nn_grid[i, :degree] = (i + torch.arange(1, degree + 1)) % n_samples

    # Global negative-sample columns, one row per point.
    neg_indices = torch.randint(
        0, n_samples, (n_samples, n_negatives), generator=generator
    )

    # Per-edge epoch schedule aligned to the padded grid: even columns are due
    # this step (active), odd columns are not, so every row keeps >= 1 active
    # attractive edge and the per-row negative-count filter stays non-trivial.
    columns = torch.arange(max_degree)
    epoch_grid = (
        torch.where(columns % 2 == 0, 1.0, 5.0)
        .to(torch.float64)
        .unsqueeze(0)
        .expand(n_samples, max_degree)
        .contiguous()
    )
    return embedding, nn_grid, neg_indices, epoch_grid


def _gradients_for_rows(rows, embedding, nn_grid, neg_indices, epoch_grid):
    """Closed-form UMAP gradients for the chunk of global rows ``rows``.

    Builds exactly the buffers the gradient methods read, flattening the chunk's
    slice of the padded grid the way ``on_affinity_computation_end`` does, and
    returns ``(attractive, repulsive)``.
    """
    grid = nn_grid[rows]
    source, target, mask = UMAP._flatten_padded_edges(grid)

    model = UMAP(n_components=embedding.shape[1], n_neighbors=2, optimizer="SGD")
    model._a, model._b, model._eps = 1.577, 0.895, 1e-3
    model.n_iter_ = 0
    model.negative_sample_rate = 1
    model.n_negatives = neg_indices.shape[1]

    model.embedding_ = embedding  # replicated on every rank
    model.chunk_indices_ = rows  # global row ids owned by this rank
    model.attractive_source_ = source  # local edge source (0..len(rows)-1)
    model.attractive_target_ = target  # global neighbor column
    model.attractive_counts_ = mask.sum(dim=1)
    model.epoch_of_next_sample = epoch_grid[rows][mask].clone()
    model.epochs_per_sample = torch.ones_like(model.epoch_of_next_sample)
    model.neg_indices_ = neg_indices[rows]  # global negative columns

    attractive = model._compute_attractive_gradients()
    repulsive = model._compute_repulsive_gradients()  # reads mask_affinity_in_
    return attractive, repulsive


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_umap_chunk_gradients_match_full_slice(world_size):
    """Each rank's chunk gradient equals the full-data gradient's slice."""
    embedding, nn_grid, neg_indices, epoch_grid = _make_umap_state()
    n_samples = embedding.shape[0]

    full_attractive, full_repulsive = _gradients_for_rows(
        torch.arange(n_samples), embedding, nn_grid, neg_indices, epoch_grid
    )
    # The reference gradients must be non-trivial, otherwise a wrong-row
    # regression could not change the compared values.
    assert full_attractive.abs().sum() > 1e-6
    assert full_repulsive.abs().sum() > 1e-6

    for rows in torch.chunk(torch.arange(n_samples), world_size):
        chunk_attractive, chunk_repulsive = _gradients_for_rows(
            rows, embedding, nn_grid, neg_indices, epoch_grid
        )
        torch.testing.assert_close(
            chunk_attractive, full_attractive[rows], rtol=1e-9, atol=1e-9
        )
        torch.testing.assert_close(
            chunk_repulsive, full_repulsive[rows], rtol=1e-9, atol=1e-9
        )
