"""End-to-end distributed neighbor-embedding tests on real GPUs (NCCL).

Unlike the transport-agnostic pieces in
``test_distributed_neighbor_embedding.py`` (which run on the Gloo CPU backend),
the full ``UMAP``/``InfoTSNE`` distributed path is GPU-only by design: both
:class:`torchdr.affinity.base.SparseAffinity` and
:class:`torchdr.neighbor_embedding.base.NeighborEmbedding` force a CUDA device
in distributed mode and raise on ``device="cpu"``. These tests therefore cover
the whole stack -- k-NN sharding, distributed sparse symmetrization, and the
per-iteration gradient collective -- against a single-process reference, on a
real NCCL process group.

They are **not** wired into GitHub CI, which has no multi-GPU runner. They are
gated behind ``TORCHDR_DISTRIBUTED_GPU_TEST=1`` (plus CUDA and at least two
ranks) and are meant to be run manually or on a scheduled provisioned runner::

    TORCHDR_DISTRIBUTED_GPU_TEST=1 python -m torch.distributed.run \\
        --standalone --nnodes=1 --nproc-per-node=2 \\
        -m pytest torchdr/tests/test_distributed_neighbor_embedding_gpu.py -q
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr import InfoTSNE, UMAP
from torchdr.affinity import UMAPAffinity
from torchdr.distance import FaissPlanConfig
from torchdr.distributed import init_distributed, shutdown_distributed


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_GPU_TEST") != "1"
    or not torch.cuda.is_available(),
    reason="run manually on a provisioned NCCL multi-GPU runner",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the NCCL process group created by torchrun.

    Fails loudly on a single process: these tests are meaningless unless the
    rows really cross a rank boundary, and a silent one-rank run would report
    green while exercising none of the distributed path.
    """
    # TorchDR already creates the NCCL group on import when launched under
    # torchrun with a GPU, so go through its idempotent helper: calling
    # torch.distributed.init_process_group directly would raise "trying to
    # initialize the default process group twice". shutdown_distributed only
    # tears down a group TorchDR itself created, so it is a safe cleanup.
    init_distributed(backend="nccl")
    if not dist.is_initialized():
        pytest.fail("launch this module with torchrun and at least two processes")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    if world_size < 2:
        shutdown_distributed()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    shutdown_distributed()


def _identical_data(n_samples, n_features, dtype, seed=0):
    """Deterministic data, identical on every rank.

    The distributed design replicates the full input on each rank and shards
    only the *rows* that a rank owns, so every rank must start from the same X.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, n_features, generator=generator, dtype=dtype)


def _rowwise_to_dense(values, indices, n_columns):
    """Padded row-wise sparse representation to dense, summing duplicate slots."""
    dense = torch.zeros((values.shape[0], n_columns), dtype=values.dtype)
    valid = indices >= 0
    dense.scatter_add_(1, indices.clamp_min(0), values * valid)
    return dense


def _gather_full_matrix(local_dense):
    """All-gather each rank's row block (variable height) into the full matrix."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_dense.cpu())
    return torch.cat(gathered)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distributed_umap_affinity_matches_single_process(dtype):
    """Distributed UMAP affinity must equal the single-process affinity.

    Each rank computes the affinity for its own chunk of rows against the full
    replicated input; concatenating the chunks must rebuild exactly the matrix
    a single process assembles. This exercises the real k-NN + distributed
    ``sum_minus_prod`` symmetrization wiring and is sensitive to both the
    symmetrization-direction bug (rows would stop matching) and any dtype drift.
    """
    n_samples, n_features = 128, 16
    X = _identical_data(n_samples, n_features, dtype)

    distributed = UMAPAffinity(
        n_neighbors=15, distributed=True, backend=FaissPlanConfig()
    )
    local_values, local_indices = distributed(X.cuda())
    assert distributed.faiss_plan_.distribution == "replicate"
    local_dense = _rowwise_to_dense(local_values.cpu(), local_indices.cpu(), n_samples)
    assert local_dense.shape[0] == distributed.chunk_size_
    full = _gather_full_matrix(local_dense)

    single = UMAPAffinity(n_neighbors=15, distributed=False, backend=FaissPlanConfig())
    ref_values, ref_indices = single(X.cuda())
    assert single.faiss_plan_.distribution == "single"
    reference = _rowwise_to_dense(ref_values.cpu(), ref_indices.cpu(), n_samples)

    assert full.shape == (n_samples, n_samples)
    assert full.dtype == reference.dtype == dtype
    torch.testing.assert_close(full, reference)
    # UMAP's sum_minus_prod symmetrization is exactly symmetric; a direction
    # bug in the distributed exchange breaks this even when shapes line up.
    torch.testing.assert_close(full, full.T)


def _assert_replicated_embedding(embedding, n_samples, n_components):
    """Embedding is finite, correctly shaped, and identical on every rank."""
    assert embedding.shape == (n_samples, n_components)
    assert torch.isfinite(embedding).all()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, embedding.detach().cpu())
    for other in gathered[1:]:
        # Every rank applies the same all-reduced gradient to the same initial
        # embedding, so the replicated embeddings must stay in lock-step.
        torch.testing.assert_close(gathered[0], other)


def test_distributed_umap_fit_smoke():
    """A full distributed UMAP fit (closed-form path) runs and stays replicated."""
    n_samples, n_features, n_components = 128, 16, 2
    X = _identical_data(n_samples, n_features, torch.float32, seed=1)

    model = UMAP(
        n_neighbors=15,
        n_components=n_components,
        max_iter=50,
        random_state=0,
        distributed=True,
    )
    embedding = model.fit_transform(X.cuda())

    _assert_replicated_embedding(embedding, n_samples, n_components)


def test_distributed_infotsne_fit_smoke():
    """A full distributed InfoTSNE fit (autograd path) runs and stays replicated."""
    n_samples, n_features, n_components = 128, 16, 2
    X = _identical_data(n_samples, n_features, torch.float32, seed=2)

    model = InfoTSNE(
        perplexity=30,
        n_components=n_components,
        max_iter=50,
        random_state=0,
        distributed=True,
    )
    embedding = model.fit_transform(X.cuda())

    _assert_replicated_embedding(embedding, n_samples, n_components)
