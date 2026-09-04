"""Real-process coverage for distributed PCA aggregation."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr import PCA
from torchdr.distributed import DistributedContext


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
    reason="run through the dedicated multi-process integration workflow",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the Gloo group created by torchrun."""
    dist.init_process_group(backend="gloo")
    if dist.get_world_size() < 2:
        dist.destroy_process_group()
        pytest.fail("launch this module with at least two processes")
    yield
    dist.destroy_process_group()


def test_distributed_pca_matches_full_data_reference():
    """Global statistics and projections agree across an uneven row split."""
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(101, 6, dtype=torch.float64, generator=generator)
    X.mul_(torch.arange(1, 7, dtype=X.dtype))
    n_components = 4

    reference = PCA(n_components=n_components, distributed=False)
    reference_embedding = reference._fit_transform_standard(X)

    context = DistributedContext()
    start, end = context.compute_chunk_bounds(len(X))
    model = PCA(n_components=n_components, distributed=True)
    local_embedding = model._fit_transform_distributed(X[start:end].contiguous())

    # Complete the test's only extra collective before asserting, so a failed
    # comparison cannot strand a peer in communication during test teardown.
    gathered = [torch.empty_like(model.components_) for _ in range(context.world_size)]
    dist.all_gather(gathered, model.components_)

    torch.testing.assert_close(model.mean_, reference.mean_, rtol=1e-8, atol=1e-8)

    # PCA axes are unique only up to sign. Use the component alignment for both
    # the axes and their corresponding projection columns.
    signs = torch.sign((model.components_ * reference.components_).sum(dim=1))
    signs.masked_fill_(signs == 0, 1)
    torch.testing.assert_close(
        model.components_ * signs[:, None],
        reference.components_,
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        local_embedding * signs[None, :],
        reference_embedding[start:end],
        rtol=1e-8,
        atol=1e-8,
    )

    for components in gathered[1:]:
        torch.testing.assert_close(components, gathered[0], rtol=0, atol=0)
