"""Real-process tests for training a FAISS index once for a group of ranks.

These run on the Gloo CPU backend against a CPU FAISS index, so ordinary CI
runners cover the broadcast that gives every rank the same trained quantizer.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

from torchdr.distance import FaissConfig, pairwise_distances
from torchdr.distance.faiss import (
    _build_index_from_dataloader,
    _create_index,
    _train_index,
)
from torchdr.distributed import DistributedContext
from torchdr.utils import faiss


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
        reason="run through the dedicated multi-process integration workflow",
    ),
    pytest.mark.skipif(faiss is None or faiss is False, reason="faiss not installed"),
]

N_SAMPLES = 4000
N_FEATURES = 8
NLIST = 16
METRIC = "sqeuclidean"


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the process group created by torchrun.

    Fails loudly on a single process: one rank trains for itself, so a silent
    one-rank run would report green while exercising none of the broadcast.
    """
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    if world_size < 2:
        dist.destroy_process_group()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def context(distributed_process_group):
    return DistributedContext()


@pytest.fixture(scope="module")
def data():
    """The same dataset on every rank, which is the contract TorchDR expects."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N_SAMPLES, N_FEATURES, generator=generator)


def _untrained(index_type="IVF", **kwargs):
    config = FaissConfig(index_type=index_type, nlist=NLIST, **kwargs)
    return _create_index(METRIC, config, N_FEATURES, N_SAMPLES, False), config


def _gather(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


class TestSharedTraining:
    def test_every_rank_ends_with_the_same_trained_index(self, data, context):
        # Each rank draws different training rows, so agreeing afterwards can
        # only come from the broadcast and not from a coincidence.
        calls = []

        def rows():
            calls.append(1)
            generator = torch.Generator().manual_seed(context.rank)
            drawn = torch.randperm(N_SAMPLES, generator=generator)[: N_SAMPLES // 2]
            return data[drawn].numpy()

        index, config = _untrained()

        index = _train_index(index, rows, config, N_FEATURES, False, context)

        assert index.is_trained
        payloads = _gather(bytes(faiss.serialize_index(index)))
        assert len(set(payloads)) == 1
        assert len(calls) == (1 if context.rank == 0 else 0)
        assert _gather(len(calls)).count(1) == 1

    def test_a_failure_on_the_first_rank_reaches_every_rank(self, data, context):
        def rows():
            raise ValueError("no training rows available")

        index, config = _untrained()
        with pytest.raises(RuntimeError, match="no training rows available"):
            _train_index(index, rows, config, N_FEATURES, False, context)

    def test_a_flat_index_never_reaches_the_broadcast(self, data, context):
        # Flat has nothing to train, so the ranks skip the collective entirely
        # and each still answers its own chunk exactly.
        index, _ = _untrained(index_type="Flat")
        assert index.is_trained

        _, exact = pairwise_distances(
            data, k=5, backend=FaissConfig(), return_indices=True
        )
        _, chunk = pairwise_distances(
            data,
            k=5,
            backend=FaissConfig(),
            return_indices=True,
            distributed_ctx=context,
        )

        start, end = context.compute_chunk_bounds(N_SAMPLES)
        assert torch.equal(chunk, exact[start:end])

    # Probing every list makes IVF exact, so only ties keep it below one. IVFPQ
    # answers from its codes, and the loose bound is there to catch an index
    # that arrived broken rather than to measure how good the codes are.
    @pytest.mark.parametrize("index_type, min_recall", [("IVF", 0.99), ("IVFPQ", 0.5)])
    def test_the_received_index_still_finds_neighbors(
        self, data, context, index_type, min_recall
    ):
        config = FaissConfig(index_type=index_type, nlist=NLIST, nprobe=NLIST, M=2)
        original_config = vars(config).copy()
        _, exact = pairwise_distances(
            data, k=5, backend=FaissConfig(), return_indices=True
        )

        _, approximate = pairwise_distances(
            data, k=5, backend=config, return_indices=True, distributed_ctx=context
        )

        start, end = context.compute_chunk_bounds(N_SAMPLES)
        mine = exact[start:end]
        hits = (mine.unsqueeze(2) == approximate.unsqueeze(1)).any(dim=2).sum()
        assert hits / mine.numel() > min_recall
        assert vars(config) == original_config


class TestSharedTrainingFromDataLoader:
    def test_only_the_first_rank_streams_training_batches(self, data, context):
        loader = DataLoader(TensorDataset(data), batch_size=500, shuffle=False)
        config = FaissConfig(index_type="IVF", nlist=NLIST)
        staged = []

        def stage(batch):
            staged.append(len(batch))
            return batch.numpy()

        index, _ = _build_index_from_dataloader(
            loader, METRIC, config, stage, None, False, context
        )

        # The adding pass stages every batch; a training pass would stage more.
        assert index.is_trained
        assert index.ntotal == N_SAMPLES
        assert len(staged) == (len(loader) if context.rank else 2 * len(loader))
        assert len(set(_gather(bytes(faiss.serialize_index(index))))) == 1
