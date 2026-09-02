"""Tests for the distributed nearest-neighbor input contract."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import torchdr.distributed.input_contract as contract
from torchdr.distributed import DistributedContext
from torchdr.distributed.input_contract import validate_distributed_input


def _context(rank, world_size):
    return SimpleNamespace(
        is_initialized=True, rank=rank, world_size=world_size, local_rank=0
    )


def _loader(X, batch_size=8, **kwargs):
    return DataLoader(TensorDataset(X), batch_size=batch_size, **kwargs)


@pytest.fixture
def simulate_world(monkeypatch):
    """Substitute the metadata collective with inputs from scripted ranks."""

    def simulate(inputs, rank=0):
        monkeypatch.setattr(contract.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(contract.dist, "get_backend", lambda: "gloo")

        def fake_all_gather(output, local):
            rows = output.view(len(inputs), -1)
            for peer_rank, peer in enumerate(inputs):
                if peer_rank == rank:
                    rows[peer_rank] = local
                else:
                    rows[peer_rank] = torch.tensor(
                        contract._local_metadata(
                            peer, contract._loader_shard_info(peer) is not None
                        ),
                        dtype=torch.int64,
                    )

        monkeypatch.setattr(contract.dist, "all_gather_into_tensor", fake_all_gather)
        return validate_distributed_input(inputs[rank], _context(rank, len(inputs)))

    return simulate


class TestAcceptedInputs:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_replicated_tensors(self, simulate_world, dtype):
        X = torch.randn(64, 4).to(dtype)
        assert simulate_world([X, X.clone(), X.clone()]) is None

    def test_replicated_loaders_may_batch_differently(self, simulate_world):
        X = torch.randn(64, 4)
        assert simulate_world([_loader(X, 7), _loader(X, 13)]) is None

    def test_non_distributed_is_noop(self):
        X = torch.randn(32, 4)
        assert validate_distributed_input(X, None) is None
        assert validate_distributed_input(X, DistributedContext()) is None


class TestRejectedInputs:
    def test_distributed_sampler_is_rejected_on_every_rank(self, simulate_world):
        X = torch.randn(64, 4)
        loaders = [
            _loader(
                X,
                sampler=DistributedSampler(
                    TensorDataset(X), num_replicas=2, rank=rank, shuffle=False
                ),
            )
            for rank in range(2)
        ]

        for rank in range(2):
            with pytest.raises(ValueError, match="DistributedSampler.*32.*64"):
                simulate_world(loaders, rank)

    def test_one_sharded_rank_is_reported_to_every_rank(self, simulate_world):
        X = torch.randn(64, 4)
        inputs = [
            _loader(X),
            _loader(
                X,
                sampler=DistributedSampler(
                    TensorDataset(X), num_replicas=2, rank=1, shuffle=False
                ),
            ),
        ]

        with pytest.raises(ValueError, match=r"rank\(s\) 1"):
            simulate_world(inputs, rank=0)

    def test_drop_last_is_rejected(self, simulate_world):
        X = torch.randn(65, 4)
        loaders = [_loader(X, batch_size=8, drop_last=True) for _ in range(2)]
        with pytest.raises(ValueError, match="64.*65"):
            simulate_world(loaders)

    @pytest.mark.parametrize(
        ("inputs", "field"),
        [
            ([torch.randn(64, 4), torch.randn(63, 4)], "n_samples"),
            ([torch.randn(64, 4), torch.randn(64, 3)], "n_features"),
            ([torch.randn(64, 4), torch.randn(64, 4).double()], "dtype"),
        ],
    )
    def test_tensor_metadata_mismatch(self, simulate_world, inputs, field):
        with pytest.raises(ValueError, match=field):
            simulate_world(inputs)

    def test_input_kind_mismatch(self, simulate_world):
        X = torch.randn(64, 4)
        with pytest.raises(ValueError, match="input_kind"):
            simulate_world([X, _loader(X)])

    def test_loader_dataset_size_mismatch(self, simulate_world):
        X = torch.randn(64, 4)
        with pytest.raises(ValueError, match="n_samples"):
            simulate_world([_loader(X), _loader(X[:48])])

    def test_sharded_loader_is_rejected_without_process_group(self):
        X = torch.randn(64, 4)
        loader = _loader(
            X,
            sampler=DistributedSampler(
                TensorDataset(X), num_replicas=2, rank=0, shuffle=False
            ),
        )
        with pytest.raises(ValueError, match="DistributedSampler"):
            validate_distributed_input(loader, _context(0, 1))
