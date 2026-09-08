"""End-to-end distributed UMAP with ``input_layout='sharded'`` on real GPUs.

``test_distributed_neighbor_embedding_gpu.py`` replicates the *full* input on
every rank and shards only the embedding rows. These tests cover the orthogonal
input-sharding vertical slice of issue #359/#308: every rank holds a **distinct
contiguous shard** of the raw feature rows, the per-rank Flat index sees only
that shard, and the embedding stays replicated. This shards the ``O(N * d)``
input and index footprint across ranks while keeping the ``O(N * n_components)``
embedding on every rank.

The whole estimator path forces a CUDA device in distributed mode, so this is
GPU-only and **not** wired into GitHub CI. It is gated behind
``TORCHDR_DISTRIBUTED_GPU_TEST=1`` (plus CUDA and at least two ranks) and is run
manually or on a provisioned multi-GPU runner::

    TORCHDR_DISTRIBUTED_GPU_TEST=1 python -m torch.distributed.run \\
        --standalone --nnodes=1 --nproc-per-node=2 \\
        -m pytest torchdr/tests/test_distributed_umap_input_shard.py -q
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr import UMAP
from torchdr.affinity import UMAPAffinity
from torchdr.distance import FaissPlanConfig
from torchdr.distributed import init_distributed, shutdown_distributed
from torchdr.spectral_embedding import PCA


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_GPU_TEST") != "1"
    or not torch.cuda.is_available(),
    reason="run manually on a provisioned NCCL multi-GPU runner",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Join the NCCL group created by torchrun; fail loudly on a single rank."""
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


def _global_data(n_samples, n_features, dtype, seed=0):
    """Deterministic global dataset, identical on every rank before sharding."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, n_features, generator=generator, dtype=dtype)


def _shard_counts(n, world_size, uneven=False):
    """Rank-major shard sizes that sum to ``n`` (balanced or deliberately skewed)."""
    base, rem = divmod(n, world_size)
    counts = [base + (1 if r < rem else 0) for r in range(world_size)]
    if uneven and world_size >= 2 and counts[-1] > 1:
        shift = min(counts[-1] - 1, max(1, base // 3))
        counts[0] += shift
        counts[-1] -= shift
    return counts


def _local_shard(X_global, counts, rank):
    """The contiguous rank-major block of rows this rank owns."""
    offset = sum(counts[:rank])
    return X_global[offset : offset + counts[rank]]


def _rowwise_to_dense(values, indices, n_columns):
    """Padded row-wise sparse (over global column ids) to dense."""
    dense = torch.zeros((values.shape[0], n_columns), dtype=values.dtype)
    valid = indices >= 0
    dense.scatter_add_(1, indices.clamp_min(0), values * valid)
    return dense


def _gather_rows(local_dense):
    """All-gather each rank's (variable-height) row block into the full matrix."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_dense.cpu())
    return torch.cat(gathered)


def _assert_replicated(embedding, n_samples, n_components):
    """Embedding is finite, spans all rows, and is identical on every rank."""
    assert embedding.shape == (n_samples, n_components)
    assert torch.isfinite(embedding).all()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, embedding.detach().cpu())
    for other in gathered[1:]:
        torch.testing.assert_close(gathered[0], other)


@pytest.mark.parametrize("uneven", [False, True])
def test_input_sharded_affinity_matches_single_process(uneven):
    """Sharded-input affinity, reassembled, equals the single-process affinity.

    Each rank searches its own shard against the reconstructed global dataset and
    returns its row block with *global* column ids. Concatenating the blocks in
    rank order must rebuild exactly the matrix a single replicated process
    assembles over the same rows -- exercising the per-shard Flat search, the
    global top-k merge, and the uneven-aware distributed symmetrization.
    """
    n_samples, n_features = 160, 24
    dtype = torch.float32
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    X_global = _global_data(n_samples, n_features, dtype, seed=0)
    counts = _shard_counts(n_samples, world_size, uneven=uneven)
    assert sum(counts) == n_samples
    X_local = _local_shard(X_global, counts, rank).cuda()

    backend = FaissPlanConfig(distribution="shard") if uneven else "faiss"
    sharded = UMAPAffinity(
        n_neighbors=15,
        distributed=True,
        backend=backend,
        input_layout="sharded",
    )
    local_values, local_indices = sharded(X_local)
    assert sharded.n_global_ == n_samples
    assert local_values.shape[0] == counts[rank]
    if isinstance(backend, FaissPlanConfig):
        assert sharded.faiss_plan_.distribution == "shard"
        assert sharded.faiss_plan_.index_memory_bytes == max(counts) * n_features * 4
    local_dense = _rowwise_to_dense(local_values.cpu(), local_indices.cpu(), n_samples)
    full = _gather_rows(local_dense)

    single = UMAPAffinity(n_neighbors=15, distributed=False, backend="faiss")
    ref_values, ref_indices = single(X_global.cuda())
    reference = _rowwise_to_dense(ref_values.cpu(), ref_indices.cpu(), n_samples)

    assert full.shape == (n_samples, n_samples)
    torch.testing.assert_close(full, reference)
    # sum_minus_prod symmetrization is exactly symmetric; a routing bug in the
    # uneven-shard column exchange breaks this even when shapes line up.
    torch.testing.assert_close(full, full.T)


@pytest.mark.parametrize("uneven", [False, True])
def test_input_sharded_umap_fit_spans_global(uneven):
    """A full sharded-input UMAP fit yields a replicated global-N embedding.

    The embedding, learning rate, and negative pool must all span the global row
    count -- not the local shard -- and the replicated embedding must stay in
    lock-step across ranks (rank-0 broadcast + all-reduced gradients).
    """
    n_samples, n_features, n_components = 160, 24, 2
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    X_global = _global_data(n_samples, n_features, torch.float32, seed=1)
    counts = _shard_counts(n_samples, world_size, uneven=uneven)
    X_local = _local_shard(X_global, counts, rank).cuda()

    model = UMAP(
        n_neighbors=15,
        n_components=n_components,
        max_iter=50,
        random_state=0,
        distributed=True,
        input_layout="sharded",
        backend="faiss",
        init="random",
    )
    embedding = model.fit_transform(X_local)

    # Global-N threading: bookkeeping spans all rows, not just this shard.
    assert model.n_global_ == n_samples
    assert model.n_samples_in_ == n_samples
    assert model.chunk_indices_.numel() == counts[rank]
    if model.lr == "auto":
        assert model.lr_ == max(n_samples / model.early_exaggeration_coeff_ / 4, 50)
    _assert_replicated(embedding, n_samples, n_components)


def test_input_sharded_pca_init_matches_global_pca():
    """The default initializer uses all shards without gathering raw features."""
    n_samples, n_features, n_components = 161, 24, 2
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    X_global = _global_data(n_samples, n_features, torch.float32, seed=2).cuda()
    counts = _shard_counts(n_samples, world_size, uneven=True)
    X_local = _local_shard(X_global, counts, rank)

    model = UMAP(
        n_neighbors=15,
        n_components=n_components,
        max_iter=0,
        scheduler=None,
        distributed=True,
        input_layout="sharded",
        backend="faiss",
    )
    embedding = model.fit_transform(X_local)

    reference = PCA(
        n_components=n_components,
        distributed=False,
        process_duplicates=False,
    ).fit_transform(X_global)
    reference = model.init_scaling * reference / reference[:, 0].std()

    # PCA axes are defined up to sign. Align each coordinate before comparison.
    signs = torch.sign((embedding * reference).sum(dim=0))
    signs.masked_fill_(signs == 0, 1)
    torch.testing.assert_close(
        embedding * signs,
        reference,
        rtol=2e-4,
        atol=2e-5,
    )
    _assert_replicated(embedding, n_samples, n_components)
