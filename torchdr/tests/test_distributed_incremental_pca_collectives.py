"""Real-process coverage for distributed ExactIncrementalPCA aggregation."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr import ExactIncrementalPCA
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


def test_distributed_incremental_pca_matches_full_data_reference():
    """Aggregated statistics agree with a single-process fit over an uneven split.

    Each rank owns a disjoint chunk of rows. The all-reduced local sum and count
    form the global mean, the all-reduced covariance feeds the rank-0
    eigendecomposition, and the components and explained variance are broadcast
    from rank 0. The uneven split makes ``n_local`` differ across ranks, so a
    ``/ n_local`` mean regression cannot coincide with ``/ n_total`` and dropping
    the covariance all-reduce leaves every rank eigendecomposing only its own
    chunk -- both diverge from the single-process reference.
    """
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(101, 6, dtype=torch.float64, generator=generator)
    X.mul_(torch.arange(1, 7, dtype=X.dtype))
    n_components = 4

    reference = ExactIncrementalPCA(n_components=n_components, distributed=False)
    reference.fit(X)

    context = DistributedContext()
    start, end = context.compute_chunk_bounds(len(X))
    assert end - start < len(X), "rows must be split across ranks"
    model = ExactIncrementalPCA(n_components=n_components, distributed=True)
    model.fit(X[start:end].contiguous())

    # Complete the test's only extra collective before asserting, so a failed
    # comparison cannot strand a peer in communication during test teardown.
    gathered = [torch.empty_like(model.components_) for _ in range(context.world_size)]
    dist.all_gather(gathered, model.components_)

    # Sample count: dropping the count all-reduce collapses n_total to n_local.
    assert model.n_samples_seen_ == reference.n_samples_seen_

    # Global mean: catches dividing by the local count or dropping the sum
    # all-reduce, both of which rescale the mean.
    torch.testing.assert_close(model.mean_, reference.mean_, rtol=1e-8, atol=1e-8)

    # Principal axes are unique only up to sign; align before comparing so a
    # covariance mismatch is what fails the assertion, not the sign convention.
    signs = torch.sign((model.components_ * reference.components_).sum(dim=1))
    signs.masked_fill_(signs == 0, 1)
    torch.testing.assert_close(
        model.components_ * signs[:, None],
        reference.components_,
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        model.explained_variance_, reference.explained_variance_, rtol=1e-8, atol=1e-8
    )

    # The broadcast must leave every rank with identical components.
    for components in gathered[1:]:
        torch.testing.assert_close(components, gathered[0], rtol=0, atol=0)
