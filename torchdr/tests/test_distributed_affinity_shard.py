"""Real-process oracle tests for the step-2 sparse-affinity integration.

These validate that a UMAP affinity built over an explicitly *input-sharded*
dataset -- contiguous but possibly uneven rank-major row shards, no rank holding
the whole input -- reconstructs the single-process affinity. Two layers are
covered:

* ``distributed_symmetrize_sparse`` with an explicit rank-major boundary table
  (``owner_boundaries``): the only cross-rank coupling in UMAP's otherwise
  row-local normalization. On uneven shards the balanced arithmetic of
  :meth:`torchdr.distributed.DistributedContext.get_rank_for_indices` routes
  transpose edges to the wrong owner; the boundary table fixes it.
* The full affinity path end to end: exact input-sharded Flat k-NN
  (:func:`input_sharded_pairwise_distances_faiss`, step 1 / PR #360) feeding the
  real UMAP normalization (:func:`_log_P_UMAP` + the same binary search as
  :class:`UMAPAffinity`) and the boundary-aware symmetrization. The concatenated
  per-rank result must match a single-process oracle over the reconstructed
  input.

Like ``test_distributed_sparse.py`` and ``test_distributed_input_shard.py`` these
run on the Gloo CPU backend and are launched with
``torchrun --nproc-per-node=N -m pytest`` under ``TORCHDR_DISTRIBUTED_TEST=1``.
The estimator object itself pins ``device="cuda"`` in distributed mode, so the
end-to-end estimator wiring is exercised on GPU/NCCL; here the identical
computation is driven through its component functions so the numerical
equivalence is provable on CPU.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr.affinity.knn_normalized import _log_P_UMAP
from torchdr.distance import FaissConfig, pairwise_distances
from torchdr.distance.faiss import input_sharded_pairwise_distances_faiss
from torchdr.distributed import (
    DistributedContext,
    init_distributed,
    shutdown_distributed,
)
from torchdr.distributed.input_contract import gather_shard_layout
from torchdr.utils import binary_search, faiss, kmin
from torchdr.utils.sparse import distributed_symmetrize_sparse, symmetrize_sparse


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
        reason="run through the dedicated multi-process integration workflow",
    ),
    pytest.mark.skipif(faiss is None or faiss is False, reason="faiss not installed"),
]


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the process group created by torchrun.

    Fails loudly on a single process: with one rank the whole dataset is the only
    shard, the merge never crosses a rank boundary, and the boundary-aware owner
    routing is never exercised, so a one-rank run would report green while
    testing none of the sharded path.
    """
    init_distributed()
    world_size = dist.get_world_size()
    if world_size < 2:
        shutdown_distributed()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    shutdown_distributed()


@pytest.fixture(scope="module")
def context(distributed_process_group):
    return DistributedContext()


# Uneven contiguous rank-major shard sizes per world size, all summing to 120 and
# all differing from the balanced split so the balanced owner arithmetic is wrong.
_UNEVEN_COUNTS = {
    2: [80, 40],
    3: [20, 70, 30],
    4: [15, 55, 30, 20],
}

_ORACLE_N = 120
_ORACLE_D = 8
_UMAP_N_NEIGHBORS = 10
_UMAP_MAX_ITER = 1000


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _rowwise_to_dense(values, indices, n_columns):
    """Padded row-wise representation to dense, summing duplicate slots."""
    dense = torch.zeros((values.shape[0], n_columns), dtype=values.dtype)
    valid = indices >= 0
    dense.scatter_add_(1, indices.clamp_min(0), values * valid)
    return dense


def _balanced_counts(n_samples, world_size):
    """The balanced split used by ``compute_chunk_bounds``/``chunk_bounds``."""
    base, remainder = divmod(n_samples, world_size)
    return [base + (1 if r < remainder else 0) for r in range(world_size)]


def _prefix_offsets(counts):
    """Rank-major prefix offsets of length ``world_size + 1``."""
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    return offsets


# --------------------------------------------------------------------------- #
# Layer 1: boundary-aware sparse symmetrization on uneven shards
# --------------------------------------------------------------------------- #
def _build_graph(n_samples, n_neighbors, seed, dtype):
    """Deterministic asymmetric sparse graph, identical on every rank."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0, n_samples, (n_samples, n_neighbors), generator=generator, dtype=torch.long
    )
    values = torch.rand(
        (n_samples, n_neighbors), generator=generator, dtype=torch.float64
    )
    return values.to(dtype), indices


def _run_sharded_symmetrize(counts, n_neighbors, mode, use_boundaries):
    """Symmetrize this rank's uneven shard, gather, and return (full, oracle)."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size == len(counts), "counts must match the launched world size"
    n_samples = sum(counts)

    values, indices = _build_graph(
        n_samples, n_neighbors, seed=1234, dtype=torch.float32
    )

    offsets = _prefix_offsets(counts)
    start, size = offsets[rank], counts[rank]
    boundaries = torch.tensor(offsets, dtype=torch.long) if use_boundaries else None

    out_values, out_indices = distributed_symmetrize_sparse(
        values[start : start + size].contiguous(),
        indices[start : start + size].contiguous(),
        chunk_start=start,
        chunk_size=size,
        n_total=n_samples,
        mode=mode,
        owner_boundaries=boundaries,
    )
    assert out_values.shape[0] == size

    local_dense = _rowwise_to_dense(out_values, out_indices, n_samples)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_dense)
    full = torch.cat(gathered)

    reference_values, reference_indices = symmetrize_sparse(values, indices, mode=mode)
    reference_dense = _rowwise_to_dense(reference_values, reference_indices, n_samples)
    return full, reference_dense


@pytest.mark.parametrize("mode", ["sum", "sum_minus_prod"])
def test_uneven_shards_reconstruct_reference_with_boundaries(mode):
    """Explicit boundaries make uneven shards rebuild the single-process oracle."""
    world_size = dist.get_world_size()
    counts = _UNEVEN_COUNTS.get(world_size)
    if counts is None:
        pytest.skip(f"no uneven layout defined for world_size={world_size}")

    full, reference_dense = _run_sharded_symmetrize(
        counts, n_neighbors=7, mode=mode, use_boundaries=True
    )
    assert full.shape == (sum(counts), sum(counts))
    torch.testing.assert_close(full, reference_dense)
    torch.testing.assert_close(full, full.T)


def test_balanced_fallback_is_wrong_for_uneven_shards():
    """Without boundaries the balanced owner arithmetic drops transpose edges.

    Proves the boundary table is non-vacuous: on an uneven layout the balanced
    fallback routes some transpose edges to the wrong owner, which then discards
    them, yielding an asymmetric matrix that disagrees with the oracle.
    """
    world_size = dist.get_world_size()
    counts = _UNEVEN_COUNTS.get(world_size)
    if counts is None:
        pytest.skip(f"no uneven layout defined for world_size={world_size}")

    full, reference_dense = _run_sharded_symmetrize(
        counts, n_neighbors=7, mode="sum_minus_prod", use_boundaries=False
    )
    assert not torch.allclose(full, reference_dense), (
        "balanced fallback unexpectedly matched the oracle on an uneven layout"
    )


def test_boundaries_owner_matches_balanced_when_layout_is_balanced():
    """The boundary lookup is a strict generalization of the balanced formula."""
    for n_samples, world_size in [(120, 2), (120, 3), (120, 4), (101, 7), (15, 3)]:
        boundaries = torch.tensor(
            _prefix_offsets(_balanced_counts(n_samples, world_size)), dtype=torch.long
        )
        idx = torch.arange(n_samples)
        balanced = DistributedContext.get_rank_for_indices(idx, n_samples, world_size)
        by_boundaries = DistributedContext.get_rank_for_indices_from_boundaries(
            idx, boundaries
        )
        assert torch.equal(balanced, by_boundaries), (
            f"mismatch at n_samples={n_samples}, world_size={world_size}"
        )


def test_boundaries_owner_handles_uneven_layout_directly():
    """The canonical uneven example from the design map resolves owners exactly."""
    boundaries = torch.tensor([0, 5, 12, 15], dtype=torch.long)  # shards [5, 7, 3]
    idx = torch.arange(15)
    owners = DistributedContext.get_rank_for_indices_from_boundaries(idx, boundaries)
    expected = torch.tensor([0] * 5 + [1] * 7 + [2] * 3)
    assert torch.equal(owners, expected)


# --------------------------------------------------------------------------- #
# Layer 2: end-to-end UMAP affinity over an input-sharded dataset
# --------------------------------------------------------------------------- #
def _umap_affinity(C_, n_neighbors, max_iter=_UMAP_MAX_ITER):
    """Row-local UMAP normalization.

    Byte-faithful to :meth:`UMAPAffinity._compute_sparse_log_affinity`: per-row
    ``rho`` is the nearest distance, ``eps`` is found so each row's affinity mass
    equals ``log2(n_neighbors)``, and the affinity is ``exp(-(C - rho) / eps)``.
    The computation is entirely row-local, so a correct sharded k-NN yields the
    same pre-symmetrization affinity per row as the single-process path.
    """
    rho = kmin(C_, k=1, dim=1)[0].squeeze().contiguous()
    dtype, device = C_.dtype, C_.device
    log_n_neighbors = torch.log2(torch.tensor(n_neighbors, dtype=dtype, device=device))

    def marginal_gap(eps):
        log_marg = _log_P_UMAP(C_, rho, eps).logsumexp(1)
        return log_marg.exp().squeeze() - log_n_neighbors

    eps = binary_search(
        f=marginal_gap,
        n=C_.shape[0],
        max_iter=max_iter,
        dtype=dtype,
        device=device,
    )
    return _log_P_UMAP(C_, rho, eps).exp()


def _global_data(n_samples, d, seed):
    """Reconstructed global input, identical on every rank via the seed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, d, generator=generator, dtype=torch.float32)


def _run_sharded_umap_oracle(counts, n_neighbors, d, seed, use_boundaries):
    """Build the sharded UMAP affinity, gather it, and return (full, oracle).

    The sharded arm runs the real component functions in the same order as the
    distributed estimator: exact input-sharded Flat k-NN over ``X_local`` giving
    global ids, the row-local UMAP normalization, then boundary-aware
    symmetrization. The oracle is the single-process affinity over the
    reconstructed input.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size == len(counts), "counts must match the launched world size"
    ctx = DistributedContext()
    n_samples = sum(counts)

    X_full = _global_data(n_samples, d, seed)
    offsets = _prefix_offsets(counts)
    start, size = offsets[rank], counts[rank]
    X_local = X_full[start : start + size].contiguous()

    # Step 1 (#360): exact k-NN over the split input, returning global ids.
    C_, indices = input_sharded_pairwise_distances_faiss(
        X_local,
        k=n_neighbors,
        metric="sqeuclidean",
        exclude_diag=True,
        distributed_ctx=ctx,
    )
    affinity = _umap_affinity(C_, n_neighbors)

    # Step 2: boundary-aware symmetrization. The layout produced by #360 is the
    # source of the global sample count, this rank's offset/size, and the
    # rank-major owner boundaries.
    layout = gather_shard_layout(X_local, ctx)
    boundaries = (
        torch.tensor(_prefix_offsets(list(layout.counts)), dtype=torch.long)
        if use_boundaries
        else None
    )
    aff_sym, idx_sym = distributed_symmetrize_sparse(
        values=affinity,
        indices=indices,
        chunk_start=layout.local_offset,
        chunk_size=layout.local_count,
        n_total=layout.global_count,
        mode="sum_minus_prod",
        owner_boundaries=boundaries,
    )
    assert aff_sym.shape[0] == size

    local_dense = _rowwise_to_dense(aff_sym, idx_sym, n_samples)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_dense)
    full = torch.cat(gathered)

    # Single-process oracle: exact Flat k-NN over the reconstructed input, the
    # identical normalization, and local symmetrization.
    C_full, idx_full = pairwise_distances(
        X_full,
        k=n_neighbors,
        metric="sqeuclidean",
        backend=FaissConfig(),
        exclude_diag=True,
        return_indices=True,
    )
    P_full = _umap_affinity(C_full, n_neighbors)
    ref_v, ref_i = symmetrize_sparse(P_full, idx_full, mode="sum_minus_prod")
    reference_dense = _rowwise_to_dense(ref_v, ref_i, n_samples)
    return full, reference_dense


class TestUMAPAffinityShardOracle:
    """The full UMAP affinity over a sharded input equals the single-process one."""

    @pytest.mark.parametrize("layout", ["even", "uneven"])
    def test_sharded_umap_affinity_matches_single_process(self, context, layout):
        world_size = dist.get_world_size()
        if layout == "even":
            counts = _balanced_counts(_ORACLE_N, world_size)
        else:
            counts = _UNEVEN_COUNTS.get(world_size)
            if counts is None:
                pytest.skip(f"no uneven layout defined for world_size={world_size}")

        full, reference = _run_sharded_umap_oracle(
            counts, _UMAP_N_NEIGHBORS, _ORACLE_D, seed=7, use_boundaries=True
        )
        assert full.shape == (sum(counts), sum(counts))
        torch.testing.assert_close(full, reference, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(full, full.T, rtol=1e-3, atol=1e-4)

    def test_uneven_umap_affinity_wrong_without_boundaries(self, context):
        """Non-vacuity: the balanced fallback breaks the end-to-end affinity too."""
        world_size = dist.get_world_size()
        counts = _UNEVEN_COUNTS.get(world_size)
        if counts is None:
            pytest.skip(f"no uneven layout defined for world_size={world_size}")

        full, reference = _run_sharded_umap_oracle(
            counts, _UMAP_N_NEIGHBORS, _ORACLE_D, seed=7, use_boundaries=False
        )
        assert not torch.allclose(full, reference, rtol=1e-3, atol=1e-4), (
            "balanced fallback unexpectedly matched the oracle on an uneven layout"
        )
