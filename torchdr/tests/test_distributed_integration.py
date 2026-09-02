"""Real-process integration tests for distributed collectives."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr.distributed import (
    DistributedContext,
    init_distributed,
    shutdown_distributed,
)
from torchdr.spectral_embedding import PCA


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
    reason="run through the dedicated multi-process integration workflow",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Join the process group started by the launcher.

    TorchDR already creates a NCCL group on import when the launcher provides
    GPUs, so the group must not be created a second time here. On CPU runners
    the automatic setup is skipped and this creates the Gloo group itself.
    """
    init_distributed(backend="gloo")
    if not dist.is_initialized():
        pytest.skip("no process group; launch with torch.distributed.run")
    yield
    dist.barrier()
    shutdown_distributed()


def _collective_device():
    """Return a device the active backend can communicate on."""
    return "cuda" if dist.get_backend() == "nccl" else "cpu"


def test_real_distributed_collectives():
    """Exercise context discovery and PCA collectives across real processes."""
    context = DistributedContext()
    device = _collective_device()

    assert context.is_initialized
    assert context.world_size == 2
    assert context.rank in (0, 1)
    assert context.local_rank == context.rank

    full_data = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 1.5, 2.5, 4.0],
            [2.0, 0.5, 3.0, 5.0],
            [3.0, 2.0, 3.5, 6.0],
            [4.0, 3.0, 5.0, 7.0],
            [5.0, 2.5, 6.0, 8.0],
            [6.0, 4.0, 7.5, 9.0],
            [7.0, 5.0, 8.0, 10.0],
        ],
        dtype=torch.float64,
    ).to(device)
    local_data = full_data.chunk(context.world_size)[context.rank].contiguous()

    distributed_pca = PCA(n_components=2, distributed=True, device=device)
    embedding = distributed_pca.fit_transform(local_data)

    assert embedding.shape == (len(local_data), 2)
    assert distributed_pca._n_samples_total == len(full_data)
    torch.testing.assert_close(
        distributed_pca.mean_, full_data.mean(dim=0, keepdim=True)
    )

    gathered_components = [
        torch.zeros_like(distributed_pca.components_) for _ in range(context.world_size)
    ]
    dist.all_gather(gathered_components, distributed_pca.components_)
    torch.testing.assert_close(gathered_components[0], gathered_components[1])

    reference_pca = PCA(n_components=2, distributed=False, device=device)
    reference_pca.fit(full_data)
    distributed_projector = distributed_pca.components_.T @ distributed_pca.components_
    reference_projector = reference_pca.components_.T @ reference_pca.components_
    torch.testing.assert_close(distributed_projector, reference_projector)
