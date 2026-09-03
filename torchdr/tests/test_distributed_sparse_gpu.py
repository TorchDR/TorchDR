"""Opt-in GPU/NCCL tests for the distributed sparse symmetrization coalesce.

These require CUDA and an NCCL process group with at least two ranks, so they
never run on the CPU-only CI. Launch them explicitly on a multi-GPU node::

    TORCHDR_DISTRIBUTED_GPU_TEST=1 python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=2 \
        -m pytest torchdr/tests/test_distributed_sparse_gpu.py -q

They cover what the Gloo CPU tests cannot: that coalescing on the accelerator
(``coalesce_device="gpu"`` / ``"auto"``) is bitwise-identical to the CPU offload
(``coalesce_device="cpu"``), and that ``"auto"`` falls back to CPU on an
accelerator out-of-memory error while ``"gpu"`` propagates it.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

import torchdr.utils.sparse as sparse_mod
from torchdr.distributed import DistributedContext
from torchdr.utils.sparse import distributed_symmetrize_sparse


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_GPU_TEST") != "1"
    or not torch.cuda.is_available(),
    reason="requires CUDA and the opt-in multi-GPU symmetrization workflow",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the NCCL group created by torchrun on this GPU node.

    Fails loudly on a single process: the exchange is meaningless unless the
    edges cross a rank boundary.
    """
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    if world_size < 2:
        dist.destroy_process_group()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    yield
    dist.barrier()
    dist.destroy_process_group()


def _local_device():
    return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))


def _build_graph(n_samples, n_neighbors, seed, dtype):
    """Deterministic asymmetric sparse graph, identical on every rank.

    Columns may repeat within a row, exercising duplicate-edge coalescing.
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0, n_samples, (n_samples, n_neighbors), generator=generator, dtype=torch.long
    )
    values = torch.rand(
        (n_samples, n_neighbors), generator=generator, dtype=torch.float64
    )
    return values.to(dtype), indices


def _build_graph_distinct(n_samples, n_neighbors, seed, dtype):
    """Graph with distinct columns per row, mirroring a real k-NN graph.

    With no repeated column in a row, every coalesced (i, j) slot receives at
    most one contribution from P and one from Pᵀ, so the CPU and accelerator
    ``scatter_add_`` are bitwise-identical (the accelerator's non-deterministic
    atomic accumulation only bites for three or more summands per slot).
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.rand((n_samples, n_samples), generator=generator).argsort(dim=1)
    indices = order[:, :n_neighbors].contiguous()
    values = torch.rand(
        (n_samples, n_neighbors), generator=generator, dtype=torch.float64
    )
    return values.to(dtype), indices


def _local_chunk(values, indices, n_samples, device):
    """This rank's contiguous row chunk, moved to ``device``."""
    context = DistributedContext()
    chunk_start, chunk_end = context.compute_chunk_bounds(n_samples)
    assert chunk_end - chunk_start < n_samples, "the rows must be split across ranks"
    values_chunk = values[chunk_start:chunk_end].contiguous().to(device)
    indices_chunk = indices[chunk_start:chunk_end].contiguous().to(device)
    return chunk_start, chunk_end, values_chunk, indices_chunk


def _run(
    values_chunk, indices_chunk, chunk_start, chunk_size, n_total, coalesce_device
):
    """Symmetrize a fresh copy of the chunk with the given coalesce device."""
    return distributed_symmetrize_sparse(
        values_chunk.clone(),
        indices_chunk.clone(),
        chunk_start=chunk_start,
        chunk_size=chunk_size,
        n_total=n_total,
        mode="sum_minus_prod",
        coalesce_device=coalesce_device,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n_samples, n_neighbors", [(256, 8), (301, 7)], ids=["even", "uneven"]
)
def test_coalesce_device_bitwise_equivalence(n_samples, n_neighbors, dtype):
    """CPU, GPU, and auto coalescing yield bitwise-identical output on CUDA."""
    device = _local_device()
    values, indices = _build_graph_distinct(
        n_samples, n_neighbors, seed=n_samples, dtype=dtype
    )
    chunk_start, chunk_end, values_chunk, indices_chunk = _local_chunk(
        values, indices, n_samples, device
    )
    chunk_size = chunk_end - chunk_start

    outputs = {
        arm: _run(values_chunk, indices_chunk, chunk_start, chunk_size, n_samples, arm)
        for arm in ("cpu", "gpu", "auto")
    }

    for arm, (out_values, out_indices) in outputs.items():
        assert out_values.device.type == "cuda", arm
        assert out_indices.device.type == "cuda", arm
        assert out_values.dtype == dtype, arm

    ref_values, ref_indices = outputs["cpu"]
    for arm in ("gpu", "auto"):
        arm_values, arm_indices = outputs[arm]
        assert torch.equal(arm_indices, ref_indices), f"{arm} indices differ from cpu"
        max_abs = (arm_values - ref_values).abs().max().item()
        assert max_abs == 0.0, f"{arm} values differ from cpu by {max_abs}"


def _patch_oom_on_cuda(monkeypatch):
    """Force the coalesce merge to OOM on CUDA inputs but succeed on CPU."""
    real_merge = sparse_mod._merge_sparse_keys

    def fake_merge(keys_P, values_P, keys_PT, values_PT, n):
        if keys_P.is_cuda:
            raise torch.cuda.OutOfMemoryError
        return real_merge(keys_P, values_P, keys_PT, values_PT, n)

    monkeypatch.setattr(sparse_mod, "_merge_sparse_keys", fake_merge)


def test_auto_falls_back_to_cpu_on_oom(monkeypatch):
    """``auto`` retries on CPU after an accelerator OOM and stays correct.

    The reference and the fallback both coalesce on CPU, so the result is
    bitwise-identical regardless of duplicate edges.
    """
    n_samples, n_neighbors = 301, 7
    device = _local_device()
    values, indices = _build_graph(n_samples, n_neighbors, seed=23, dtype=torch.float32)
    chunk_start, chunk_end, values_chunk, indices_chunk = _local_chunk(
        values, indices, n_samples, device
    )
    chunk_size = chunk_end - chunk_start

    ref_values, ref_indices = _run(
        values_chunk, indices_chunk, chunk_start, chunk_size, n_samples, "cpu"
    )

    _patch_oom_on_cuda(monkeypatch)
    out_values, out_indices = _run(
        values_chunk, indices_chunk, chunk_start, chunk_size, n_samples, "auto"
    )

    assert out_values.device.type == "cuda"
    assert torch.equal(out_indices, ref_indices)
    assert (out_values - ref_values).abs().max().item() == 0.0


def test_gpu_coalesce_reraises_oom(monkeypatch):
    """``gpu`` has no fallback: an accelerator OOM propagates to the caller."""
    n_samples, n_neighbors = 301, 7
    device = _local_device()
    values, indices = _build_graph(n_samples, n_neighbors, seed=29, dtype=torch.float32)
    chunk_start, chunk_end, values_chunk, indices_chunk = _local_chunk(
        values, indices, n_samples, device
    )
    chunk_size = chunk_end - chunk_start

    _patch_oom_on_cuda(monkeypatch)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _run(values_chunk, indices_chunk, chunk_start, chunk_size, n_samples, "gpu")
