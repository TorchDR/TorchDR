"""Real-process tests for the distributed neighbor-embedding training step.

These run on the Gloo CPU backend and exercise the transport-agnostic pieces
of the sparse neighbor-embedding distributed path -- the row partitioning and
the per-iteration gradient synchronization in
:meth:`torchdr.affinity_matcher.AffinityMatcher._training_step` -- across real
processes, without a GPU or an NCCL process group.

The end-to-end ``UMAP(distributed=True).fit`` path itself is GPU-only by design
(:class:`torchdr.affinity.base.SparseAffinity` and
:class:`torchdr.neighbor_embedding.base.NeighborEmbedding` force a CUDA device
in distributed mode), so it is covered separately by
``test_distributed_neighbor_embedding_gpu.py`` on a provisioned NCCL runner.
What is verified here is the collective that every distributed neighbor
embedding relies on: each rank scatters the gradient for its own chunk of rows
into a full-size buffer and an ``all_reduce`` sums the disjoint chunks so that
every rank ends the step with the complete gradient. Both gradient modes are
covered -- the closed-form path used by UMAP and the autograd path used by
InfoTSNE, TSNE and LargeVis -- each against a single-process reference.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr import InfoTSNE, UMAP
from torchdr.distributed import DistributedContext


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
    reason="run through the dedicated multi-process integration workflow",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the process group created by torchrun.

    Fails loudly on a single process: these tests are meaningless unless the
    gradient chunks really cross a rank boundary, and a silent one-rank run
    would report green while exercising none of the collective.
    """
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    if world_size < 2:
        dist.destroy_process_group()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    dist.destroy_process_group()


def _reference_gradient(n_samples, n_components):
    """Full-matrix gradient a single process would assemble.

    Row ``i`` carries a value that depends only on its global index, so the
    reference is independent of how the rows are partitioned across ranks.
    """
    rows = torch.arange(n_samples, dtype=torch.float32)
    columns = torch.arange(1, n_components + 1, dtype=torch.float32)
    return rows[:, None] * columns[None, :]


def test_partition_tiles_the_rows_exactly_once():
    """Every global row is owned by exactly one rank, with no gaps.

    ``compute_chunk_bounds`` and ``get_rank_for_indices`` drive both the sparse
    symmetrization routing and the gradient scatter, so their agreement across
    real ranks is a precondition for everything else in this module.
    """
    context = DistributedContext()
    world_size = context.world_size

    for n_samples in (12, 97, 100):
        chunk_start, chunk_end = context.compute_chunk_bounds(n_samples)

        # This rank's chunk is non-empty and lies inside the range.
        assert 0 <= chunk_start < chunk_end <= n_samples
        assert chunk_end - chunk_start < n_samples, "rows must be split across ranks"

        # Every global index is attributed back to the rank that owns it.
        all_indices = torch.arange(n_samples)
        owners = DistributedContext.get_rank_for_indices(
            all_indices, n_samples, world_size
        )
        mine = all_indices[owners == context.rank]
        assert torch.equal(mine, torch.arange(chunk_start, chunk_end))

        # Ranks agree on a partition that covers the range exactly once.
        gathered_bounds = [None] * world_size
        dist.all_gather_object(gathered_bounds, (chunk_start, chunk_end))
        covered = sorted(gathered_bounds)
        assert covered[0][0] == 0
        assert covered[-1][1] == n_samples
        for (_, prev_end), (next_start, _) in zip(covered, covered[1:]):
            assert prev_end == next_start


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_closed_form_training_step_matches_single_process(dtype):
    """UMAP's distributed closed-form step reproduces the full gradient.

    Each rank computes the gradient for its own chunk of rows; the step
    scatters it at ``chunk_start_`` and all-reduces so every rank recovers the
    gradient a single process would have assembled from all rows.
    """
    context = DistributedContext()
    n_samples, n_components = 40, 3
    chunk_start, chunk_end = context.compute_chunk_bounds(n_samples)

    # distributed=False avoids the constructor forcing a CUDA device while a
    # process group is live; world_size>1 is set by hand to take the
    # distributed branch of _training_step, mirroring the single-process unit
    # test in test_distributed.py but with a *real* collective.
    model = UMAP(n_components=n_components, optimizer="SGD", lr=0.0, distributed=False)
    model.rank = context.rank
    model.world_size = context.world_size
    model.encoder = None
    model.scheduler_ = None
    model.device_ = torch.device("cpu")
    model.embedding_ = torch.nn.Parameter(
        torch.zeros(n_samples, n_components, dtype=dtype)
    )
    model.optimizer_ = torch.optim.SGD([model.embedding_], lr=0.0)
    model.chunk_indices_ = torch.arange(chunk_start, chunk_end)
    model.chunk_start_ = chunk_start

    reference = _reference_gradient(n_samples, n_components).to(dtype)
    local_gradient = reference[chunk_start:chunk_end].clone()
    model._compute_gradients = lambda: local_gradient

    model._training_step()

    assert model.embedding_.grad is not None
    assert model.embedding_.grad.dtype == dtype
    torch.testing.assert_close(model.embedding_.grad, reference)


def test_autograd_training_step_reduces_gradients_across_ranks():
    """InfoTSNE's distributed autograd step sums per-rank gradients.

    The autograd branch back-propagates a local loss and all-reduces
    ``embedding_.grad``. Each rank contributes a distinct scalar multiple of
    the same embedding, so the summed gradient is a deterministic multiple that
    is independent of the world size.
    """
    context = DistributedContext()
    n_samples, n_components = 16, 2

    model = InfoTSNE(
        n_components=n_components, optimizer="SGD", lr=0.0, distributed=False
    )
    model.rank = context.rank
    model.world_size = context.world_size
    model.encoder = None
    model.scheduler_ = None
    model.device_ = torch.device("cpu")
    model._use_closed_form_gradients = False
    model.embedding_ = torch.nn.Parameter(torch.ones(n_samples, n_components))
    model.optimizer_ = torch.optim.SGD([model.embedding_], lr=0.0)

    # Rank r contributes a loss of (r + 1) * sum(Z ** 2); the gradient is
    # 2 * (r + 1) * Z, so the all-reduced gradient is 2 * Z * sum_r (r + 1).
    weight = float(context.rank + 1)
    model._compute_loss = lambda: weight * (model.embedding_**2).sum()

    model._training_step()

    world_size = context.world_size
    weight_sum = world_size * (world_size + 1) / 2.0
    expected = 2.0 * torch.ones(n_samples, n_components) * weight_sum

    assert model.embedding_.grad is not None
    torch.testing.assert_close(model.embedding_.grad, expected)


def test_init_embedding_broadcasts_from_rank_zero():
    """Every rank starts optimization from rank zero's initialization."""
    context = DistributedContext()
    n_samples, n_components = 12, 3

    # Keep this transport-level test on CPU while exercising the real
    # multi-process branch.
    model = UMAP(n_components=n_components, init="normal", distributed=False)
    model.rank = context.rank
    model.world_size = context.world_size
    model.encoder = None
    model.device_ = torch.device("cpu")

    # Distinct seeds make the test fail if the broadcast is removed.
    torch.manual_seed(1000 + context.rank)

    model._init_embedding(torch.randn(n_samples, 4))

    local = model.embedding_.detach().contiguous()
    assert local.shape == (n_samples, n_components)

    gathered = [torch.empty_like(local) for _ in range(context.world_size)]
    dist.all_gather(gathered, local)

    for rank, other in enumerate(gathered):
        torch.testing.assert_close(
            other, gathered[0], msg=f"rank {rank} diverged from rank 0's init"
        )
