"""Tests for how FAISS approximate indexes resolve and train their parameters."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchdr.distance import (
    FaissConfig,
    pairwise_distances,
    pairwise_distances_faiss_from_dataloader,
)
from torchdr.distance.faiss import (
    _create_index,
    _shares_training,
    _train_index,
    _training_sample,
)
from torchdr.distributed import DistributedContext
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


class TestResolvedParameters:
    """An automatic nlist is resolved for the index, not written into config."""

    @pytest.mark.parametrize("index_type", ["IVF", "IVFPQ"])
    def test_index_gets_the_automatic_nlist(self, index_type):
        config = FaissConfig(index_type=index_type, M=2)
        index = _create_index("sqeuclidean", config, N_FEATURES, N_SAMPLES, False)

        expected = min(int(4 * np.sqrt(N_SAMPLES)), N_SAMPLES // 40, 8192)
        assert index.nlist == expected
        assert expected != config.nlist

    @pytest.mark.parametrize("index_type", ["Flat", "IVF", "IVFPQ"])
    def test_tensor_search_leaves_config_alone(self, data, index_type):
        config = FaissConfig(index_type=index_type, M=2)
        before = repr(config)

        pairwise_distances(data, k=5, backend=config, return_indices=True)

        assert repr(config) == before

    @pytest.mark.parametrize("index_type", ["Flat", "IVF", "IVFPQ"])
    def test_dataloader_search_leaves_config_alone(self, data, index_type):
        config = FaissConfig(index_type=index_type, M=2)
        before = repr(config)
        loader = DataLoader(TensorDataset(data), batch_size=2048, shuffle=False)

        pairwise_distances_faiss_from_dataloader(loader, k=5, config=config)

        assert repr(config) == before

    def test_an_explicit_nlist_is_respected(self):
        config = FaissConfig(index_type="IVF", nlist=37)
        index = _create_index("sqeuclidean", config, N_FEATURES, N_SAMPLES, False)
        assert index.nlist == 37

    def test_unsupported_index_type_is_rejected(self):
        config = FaissConfig(index_type="HNSW")
        with pytest.raises(ValueError, match="not supported"):
            _create_index("sqeuclidean", config, N_FEATURES, N_SAMPLES, False)


class TestTrainingSample:
    def test_smaller_data_is_used_whole(self, data):
        assert _training_sample(data, len(data) + 1) is data

    def test_larger_data_is_capped(self, data):
        assert len(_training_sample(data, 128)) == 128

    def test_a_seed_selects_the_same_rows(self, data):
        np.random.seed(0)
        first = _training_sample(data, 128)
        np.random.seed(0)
        assert torch.equal(_training_sample(data, 128), first)

    def test_numpy_input_is_supported(self, data):
        sample = _training_sample(data.numpy(), 128)
        assert isinstance(sample, np.ndarray)
        assert len(sample) == 128


class TestLocalTraining:
    """Without a group of ranks to share it with, training stays where it is."""

    @pytest.mark.parametrize("distributed_ctx", [None, "uninitialized"])
    def test_index_is_trained_in_place(self, data, distributed_ctx):
        if distributed_ctx == "uninitialized":
            distributed_ctx = DistributedContext()

        config = FaissConfig(index_type="IVF", nlist=16)
        index = _create_index("sqeuclidean", config, N_FEATURES, N_SAMPLES, False)
        assert not index.is_trained

        trained = _train_index(
            index, lambda: data.numpy(), config, N_FEATURES, False, distributed_ctx
        )

        assert trained is index
        assert index.is_trained
        assert not _shares_training(distributed_ctx)


class TestSeededSearch:
    """The seed the estimator sets is enough to repeat an approximate search."""

    # nlist is set low enough that the training sample is a strict subset of the
    # data, which is what makes the draw, and so the seed, matter at all.
    @pytest.mark.parametrize("index_type", ["IVF", "IVFPQ"])
    def test_the_same_seed_gives_the_same_neighbors(self, data, index_type):
        config = FaissConfig(index_type=index_type, nlist=32, M=2)

        runs = []
        for _ in range(2):
            seed_everything(0)
            _, indices = pairwise_distances(
                data, k=5, backend=config, return_indices=True
            )
            runs.append(indices)

        assert torch.equal(runs[0], runs[1])

    def test_a_different_seed_draws_different_training_rows(self, data):
        draws = []
        for seed in (0, 1):
            seed_everything(seed)
            draws.append(_training_sample(data, N_SAMPLES // 4))

        assert not torch.equal(draws[0], draws[1])
