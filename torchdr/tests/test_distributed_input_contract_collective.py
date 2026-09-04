"""Real-process test for the distributed input-contract metadata gather.

:func:`torchdr.distributed.input_contract.validate_distributed_input` rejects
violations of the replicated-input contract -- sharded loaders and rank
disagreements on shape, dtype, or input kind -- by gathering a fixed-width
metadata vector from every rank with ``dist.all_gather_into_tensor`` and
reshaping the flat buffer to ``(world_size, _VECTOR_LEN)``.

The existing ``test_distributed_input_contract.py`` monkeypatches that
collective with a fake that hardcodes the correct row-major layout, so a real
regression in the receive-buffer size, the flat->2D reshape order, or the
metadata width is invisible to it and to any single-process run. This module
runs the guard on a real two-process Gloo group so the metadata actually
crosses a rank boundary:

* replicated identical inputs pass -- this drives the real gather and reshape
  end to end, so a wrong receive-buffer size makes the collective itself raise;
* one rank with a divergent ``n_features`` is rejected on *every* rank -- only
  a correct cross-rank gather and row-major reshape let each rank read the
  peers' metadata, so a gather that returns only local data turns the
  rejection into a silent false-negative and a transposed reshape misreports
  the field; both are invisible to the mocked test;
* a per-rank sharded DataLoader is rejected on every rank -- the sharded flag
  must travel through the same gather.

The guard selects the CPU device on the Gloo backend, so it runs under the
ordinary two-process integration workflow with no GPU.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from torchdr.distributed import DistributedContext
from torchdr.distributed.input_contract import validate_distributed_input


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
    reason="run through the dedicated multi-process integration workflow",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the process group created by torchrun.

    Fails loudly on a single process: the guard's gather reduces to a no-op at
    world_size 1, so a one-rank run would report green while never crossing a
    rank boundary.
    """
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    if world_size < 2:
        dist.destroy_process_group()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    dist.destroy_process_group()


def _replicated_tensor():
    """A tensor every rank builds identically, satisfying the contract.

    Only shape and dtype reach the metadata vector, so identical shape/dtype on
    every rank is what the guard must accept; the values are irrelevant.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randn(64, 4, generator=generator)


def test_replicated_inputs_pass_the_real_gather():
    """Identical inputs on every rank clear the guard.

    This drives the real ``all_gather_into_tensor`` and reshape to completion;
    a wrong receive-buffer size would make the collective itself raise here,
    which the mocked layout can never surface.
    """
    result = validate_distributed_input(_replicated_tensor(), DistributedContext())
    assert result is None


def test_divergent_n_features_is_rejected_on_every_rank():
    """One rank with a different feature count is caught on all ranks.

    Only rank 0 keeps four features; every other rank drops to three. Detecting
    the disagreement requires each rank to read the peers' metadata through the
    gather and reshape it row-major, so a gather that returns only local data
    would let rank 0 pass (a false-negative) and a transposed reshape would
    blame the wrong field -- neither of which the mocked collective can expose.
    """
    rank = dist.get_rank()
    n_features = 4 if rank == 0 else 3
    X = torch.randn(64, n_features)
    with pytest.raises(ValueError, match="n_features"):
        validate_distributed_input(X, DistributedContext())


def _sharded_loader():
    """A DataLoader that yields only this rank's shard of the dataset."""
    X = torch.randn(64, 4)
    sampler = DistributedSampler(
        TensorDataset(X),
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )
    return DataLoader(TensorDataset(X), sampler=sampler)


def test_sharded_loader_is_rejected_on_every_rank():
    """A per-rank sharded loader is rejected; the sharded flag crosses ranks.

    Every rank sets its own sharded flag, but the guard reads the gathered
    column to report the offending ranks, so the rejection still travels
    through the real collective rather than a local shortcut.
    """
    with pytest.raises(ValueError, match="DistributedSampler"):
        validate_distributed_input(_sharded_loader(), DistributedContext())
