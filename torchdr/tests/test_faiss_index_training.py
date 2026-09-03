"""Tests for how FAISS approximate indexes resolve and train their parameters."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch

from torchdr.distance import FaissConfig, pairwise_distances
from torchdr.distance.faiss import _create_index
from torchdr.utils import faiss, seed_everything


pytestmark = pytest.mark.skipif(
    faiss is None or faiss is False, reason="faiss not installed"
)

# Above the 10000 rows that turn on the automatic ``nlist``, and enough of them
# that the resolved value differs from the default.
N_SAMPLES = 20000
N_FEATURES = 8


@pytest.fixture(scope="module")
def data():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N_SAMPLES, N_FEATURES, generator=generator)


class TestApproximateIndexTraining:
    """Stable behavior of approximate-index initialization."""

    @pytest.mark.parametrize("index_type", ["IVF", "IVFPQ"])
    def test_index_gets_the_automatic_nlist(self, index_type):
        config = FaissConfig(index_type=index_type, M=2)
        index = _create_index("sqeuclidean", config, N_FEATURES, N_SAMPLES, False)

        expected = min(int(4 * np.sqrt(N_SAMPLES)), N_SAMPLES // 40, 8192)
        assert index.nlist == expected
        assert expected != config.nlist

    @pytest.mark.parametrize("index_type", ["IVF", "IVFPQ"])
    def test_automatic_nlist_does_not_mutate_config(self, data, index_type):
        config = FaissConfig(index_type=index_type, M=2)
        before = vars(config).copy()

        pairwise_distances(data, k=5, backend=config, return_indices=True)

        assert vars(config) == before

    @pytest.mark.parametrize("index_type", ["IVF", "IVFPQ"])
    def test_seeded_search_is_reproducible(self, data, index_type):
        # nlist is low enough that training uses a random subset. This exercises
        # the public search behavior instead of the sampling helper itself.
        config = FaissConfig(index_type=index_type, nlist=32, M=2)

        runs = []
        for _ in range(2):
            seed_everything(0)
            _, indices = pairwise_distances(
                data, k=5, backend=config, return_indices=True
            )
            runs.append(indices)

        assert torch.equal(runs[0], runs[1])
