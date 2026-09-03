"""Real-process tests for the distributed sparse symmetrization exchange.

These run on the Gloo CPU backend, which only supports the flat
``all_to_all_single`` collective. They therefore cover the edge exchange of
:func:`torchdr.utils.sparse.distributed_symmetrize_sparse` on ordinary CI
runners, without a GPU or an NCCL process group.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr.distributed import DistributedContext
from torchdr.utils.sparse import distributed_symmetrize_sparse, symmetrize_sparse


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
    reason="run through the dedicated multi-process integration workflow",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the process group created by torchrun.

    Fails loudly on a single process: these tests are meaningless unless the
    edges really cross a rank boundary, and a silent one-rank run would report
    green while exercising none of the exchange.
    """
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    if world_size < 2:
        dist.destroy_process_group()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    dist.destroy_process_group()


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


def _rowwise_to_dense(values, indices, n_columns):
    """Padded row-wise representation to dense, summing duplicate slots."""
    dense = torch.zeros((values.shape[0], n_columns), dtype=values.dtype)
    valid = indices >= 0
    dense.scatter_add_(1, indices.clamp_min(0), values * valid)
    return dense


def _symmetrize_local_chunk(values, indices, n_samples, mode, coalesce_device="auto"):
    """Run the distributed path on this rank's row chunk."""
    context = DistributedContext()
    chunk_start, chunk_end = context.compute_chunk_bounds(n_samples)
    chunk_size = chunk_end - chunk_start
    assert chunk_size < n_samples, "the rows must be split across ranks"

    out_values, out_indices = distributed_symmetrize_sparse(
        values[chunk_start:chunk_end].contiguous(),
        indices[chunk_start:chunk_end].contiguous(),
        chunk_start=chunk_start,
        chunk_size=chunk_size,
        n_total=n_samples,
        mode=mode,
        coalesce_device=coalesce_device,
    )
    return chunk_start, chunk_end, out_values, out_indices


@pytest.mark.parametrize("mode", ["sum", "sum_minus_prod"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n_samples, n_neighbors", [(64, 4), (101, 7)], ids=["even", "uneven"]
)
def test_chunk_matches_single_process_reference(n_samples, n_neighbors, dtype, mode):
    """Each rank's rows must equal the single-process symmetrization."""
    values, indices = _build_graph(n_samples, n_neighbors, seed=n_samples, dtype=dtype)

    chunk_start, chunk_end, out_values, out_indices = _symmetrize_local_chunk(
        values, indices, n_samples, mode
    )

    assert out_values.dtype == dtype
    assert out_indices.dtype == torch.int64
    assert out_values.shape[0] == chunk_end - chunk_start

    reference_values, reference_indices = symmetrize_sparse(values, indices, mode=mode)
    torch.testing.assert_close(
        _rowwise_to_dense(out_values, out_indices, n_samples),
        _rowwise_to_dense(reference_values, reference_indices, n_samples)[
            chunk_start:chunk_end
        ],
    )


def test_gathered_ranks_reconstruct_the_full_matrix():
    """Concatenating every rank's rows must rebuild the reference matrix."""
    n_samples, n_neighbors = 101, 7
    values, indices = _build_graph(n_samples, n_neighbors, seed=7, dtype=torch.float32)

    _, _, out_values, out_indices = _symmetrize_local_chunk(
        values, indices, n_samples, "sum_minus_prod"
    )
    local_dense = _rowwise_to_dense(out_values, out_indices, n_samples)

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_dense)

    reference_values, reference_indices = symmetrize_sparse(
        values, indices, mode="sum_minus_prod"
    )
    full = torch.cat(gathered)
    assert full.shape == (n_samples, n_samples)
    torch.testing.assert_close(
        full, _rowwise_to_dense(reference_values, reference_indices, n_samples)
    )
    torch.testing.assert_close(full, full.T)


def test_skewed_payload_matches_reference():
    """Very unequal per-rank payloads must exchange correctly.

    Every edge points into the rows owned by the last rank, so all other ranks
    send their whole edge list to a single peer and receive almost nothing.
    This is the uneven split-size case of the flat exchange.
    """
    n_samples, n_neighbors = 96, 5
    world_size = dist.get_world_size()
    last_chunk_start = (n_samples // world_size) * (world_size - 1)

    generator = torch.Generator().manual_seed(11)
    indices = torch.randint(
        last_chunk_start,
        n_samples,
        (n_samples, n_neighbors),
        generator=generator,
        dtype=torch.long,
    )
    values = torch.rand(
        (n_samples, n_neighbors), generator=generator, dtype=torch.float32
    )

    chunk_start, chunk_end, out_values, out_indices = _symmetrize_local_chunk(
        values, indices, n_samples, "sum_minus_prod"
    )

    reference_values, reference_indices = symmetrize_sparse(
        values, indices, mode="sum_minus_prod"
    )
    torch.testing.assert_close(
        _rowwise_to_dense(out_values, out_indices, n_samples),
        _rowwise_to_dense(reference_values, reference_indices, n_samples)[
            chunk_start:chunk_end
        ],
    )


@pytest.mark.parametrize("coalesce_device", ["auto", "cpu", "gpu"])
def test_coalesce_device_matches_reference(coalesce_device):
    """Every ``coalesce_device`` setting reproduces the reference on CPU.

    On the Gloo CPU backend all three settings coalesce on the host, so this
    only locks in that the dispatch accepts each value and leaves the result
    unchanged. GPU-vs-CPU bitwise equivalence is covered by the opt-in GPU
    module (``test_distributed_sparse_gpu.py``).
    """
    n_samples, n_neighbors = 101, 7
    values, indices = _build_graph(n_samples, n_neighbors, seed=13, dtype=torch.float32)

    chunk_start, chunk_end, out_values, out_indices = _symmetrize_local_chunk(
        values, indices, n_samples, "sum_minus_prod", coalesce_device=coalesce_device
    )

    reference_values, reference_indices = symmetrize_sparse(
        values, indices, mode="sum_minus_prod"
    )
    torch.testing.assert_close(
        _rowwise_to_dense(out_values, out_indices, n_samples),
        _rowwise_to_dense(reference_values, reference_indices, n_samples)[
            chunk_start:chunk_end
        ],
    )


def test_invalid_coalesce_device_raises():
    """An unknown ``coalesce_device`` is rejected before any collective runs."""
    n_samples, n_neighbors = 64, 4
    values, indices = _build_graph(n_samples, n_neighbors, seed=5, dtype=torch.float32)

    context = DistributedContext()
    chunk_start, chunk_end = context.compute_chunk_bounds(n_samples)
    with pytest.raises(ValueError, match="coalesce_device"):
        distributed_symmetrize_sparse(
            values[chunk_start:chunk_end].contiguous(),
            indices[chunk_start:chunk_end].contiguous(),
            chunk_start=chunk_start,
            chunk_size=chunk_end - chunk_start,
            n_total=n_samples,
            coalesce_device="bogus",
        )
