"""Regression test for distributed negative-sampling coordinates."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr import InfoTSNE, LargeVis, UMAP


NEGATIVE_SAMPLING_ESTIMATORS = [InfoTSNE, LargeVis, UMAP]
N_SAMPLES = 100
CHUNK_START = 34
CHUNK_SIZE = 33
N_NEIGHBORS = 4


def _model_on_middle_chunk(cls, exclude_neighbors):
    model = cls(
        n_components=2,
        exclude_neighbors_from_negative_sampling=exclude_neighbors,
    )
    model.rank = 1
    model.world_size = 3
    model.device_ = torch.device("cpu")
    model.n_samples_in_ = N_SAMPLES
    model.affinity_in.chunk_start_ = CHUNK_START
    model.affinity_in.chunk_size_ = CHUNK_SIZE

    rows = torch.arange(CHUNK_START, CHUNK_START + CHUNK_SIZE)
    neighbors = (rows.unsqueeze(1) + 1 + torch.arange(N_NEIGHBORS)) % N_SAMPLES
    model.NN_indices_ = neighbors
    generator = torch.Generator().manual_seed(0)
    model.affinity_in_ = torch.rand(CHUNK_SIZE, N_NEIGHBORS, generator=generator)
    model.on_affinity_computation_end()
    return model, rows, neighbors


@pytest.mark.parametrize(
    "cls", NEGATIVE_SAMPLING_ESTIMATORS, ids=lambda cls: cls.__name__
)
@pytest.mark.parametrize("exclude_neighbors", [False, True], ids=["self", "neighbors"])
def test_distributed_exclusions_use_global_coordinates(cls, exclude_neighbors):
    model, rows, neighbors = _model_on_middle_chunk(cls, exclude_neighbors)

    expected = rows.unsqueeze(1)
    if exclude_neighbors:
        expected = torch.cat((expected, neighbors), dim=1).sort(dim=1).values

    # Every rank holds the full embedding, so exclusions use global row IDs and
    # the candidate count is based on the full dataset, not the local chunk.
    assert torch.equal(model.negative_exclusion_indices_, expected)
    assert torch.equal(
        model.negative_available_counts_,
        torch.full((CHUNK_SIZE,), N_SAMPLES - expected.shape[1]),
    )
