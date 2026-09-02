"""Validation of the distributed input contract.

TorchDR's distributed neighbor-embedding path expects every rank to hold the
*same full dataset*. Each rank builds a complete FAISS index over it and is
assigned a contiguous chunk of query rows, so the neighbor indices a rank
returns are global sample ids.

When that expectation is violated the failure is silent: a sharded DataLoader
makes each rank index only its own shard, and the returned indices are local
shard positions that are still inside ``[0, n_samples)``. No bounds check can
see them. The helpers here turn those cases into an early, actionable error.

Detection is deliberately collective: every rank publishes a small description
of its own input and inspects all of them, so a violation raises on all ranks
at once instead of leaving the innocent ones waiting in the next collective.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import zlib
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

__all__ = ["validate_distributed_input"]

# Number of rows sampled for the content checksum. The checksum is exact
# integer arithmetic on a bounded, deterministic subset of rows, so its cost
# does not grow with the dataset.
_DIGEST_MAX_ROWS = 4096

# Mersenne prime. Keeps every intermediate product below 2**63 so the checksum
# is exact on int64 and independent of reduction order and device.
_DIGEST_MOD = (1 << 31) - 1

_KIND_TENSOR = 0
_KIND_DATALOADER = 1

_UNKNOWN = -1

# Fields compared for equality across ranks, in their order in the vector.
_FIELDS = ("n_samples", "n_features", "dtype", "input_kind", "data_checksum")
_SHARDED_IDX = len(_FIELDS)
_VECTOR_LEN = _SHARDED_IDX + 1

_CONTRACT = (
    "TorchDR replicates the dataset on every rank and partitions the k-NN "
    "queries itself, so every rank must be given the same full dataset and "
    "the neighbor indices it returns are global sample ids."
)


def _dtype_code(dtype) -> int:
    """Stable integer code for a dtype.

    ``hash()`` on strings is salted per process, so it cannot be used to
    compare values across ranks. CRC32 is deterministic.
    """
    return int(zlib.crc32(str(dtype).encode()))


def _loader_num_iterated(dataloader: DataLoader) -> Optional[int]:
    """Number of samples ``dataloader`` will actually yield, if knowable."""
    sampler = getattr(dataloader, "sampler", None)
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if sampler is None and batch_sampler is not None:
        sampler = getattr(batch_sampler, "sampler", None)
    if sampler is None:
        return None
    try:
        return len(sampler)
    except TypeError:
        return None


def _loader_shard_info(X) -> Optional[Tuple[int, int, str]]:
    """Return ``(n_iterated, n_total, sampler_name)`` if ``X`` is sharded."""
    if not isinstance(X, DataLoader):
        return None

    dataset = getattr(X, "dataset", None)
    if dataset is None or not hasattr(dataset, "__len__"):
        return None

    n_total = len(dataset)
    n_iterated = _loader_num_iterated(X)
    if n_iterated is None or n_iterated >= n_total:
        return None

    sampler = getattr(X, "sampler", None)
    name = type(sampler).__name__ if sampler is not None else "the sampler"
    return n_iterated, n_total, name


def _sharded_loader_message(info: Optional[Tuple[int, int, str]], ranks) -> str:
    if info is not None:
        n_iterated, n_total, name = info
        detail = (
            f"this rank's {name} yields {n_iterated} of the {n_total} samples "
            "in the dataset"
        )
    else:
        listed = ", ".join(str(r) for r in ranks)
        detail = f"the DataLoader is sharded on rank(s) {listed}"
    return (
        "[TorchDR] Distributed mode requires every rank to iterate the full "
        f"dataset, but {detail}. " + _CONTRACT + " A sharded loader makes each "
        "rank index only its own shard and return neighbor indices that are "
        "local shard positions rather than global sample ids; those indices "
        "stay inside [0, n_samples), so nothing downstream can detect them. "
        "Build the DataLoader without a DistributedSampler (shuffle=False and "
        "no sampler=...); TorchDR assigns each rank its own chunk of query "
        "rows."
    )


def _tensor_checksum(X: torch.Tensor) -> int:
    """Order-sensitive exact checksum over a bounded subset of rows.

    Values are reinterpreted as integers, so the result is exact and does not
    depend on floating-point reduction order, device, or thread count. Row and
    column positions are weighted, so a permutation of the data changes the
    checksum.
    """
    if X.numel() == 0:
        return 0
    if X.dtype.is_complex:  # pragma: no cover - unsupported upstream anyway
        return _UNKNOWN

    n_rows = X.shape[0]
    step = max(1, -(-n_rows // _DIGEST_MAX_ROWS))  # ceil division
    positions = torch.arange(0, n_rows, step, device=X.device)
    rows = X.index_select(0, positions).contiguous()

    int_dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(
        rows.element_size()
    )
    if int_dtype is None:  # pragma: no cover - no such torch dtype today
        return _UNKNOWN

    bits = rows.view(int_dtype).reshape(rows.shape[0], -1).to(torch.int64)
    bits = bits % _DIGEST_MOD

    col_w = torch.arange(1, bits.shape[1] + 1, device=bits.device, dtype=torch.int64)
    col_w = col_w % _DIGEST_MOD
    per_row = ((bits * col_w) % _DIGEST_MOD).sum(dim=1) % _DIGEST_MOD

    row_w = (positions.to(torch.int64) % _DIGEST_MOD) + 1
    total = ((per_row * row_w) % _DIGEST_MOD).sum() % _DIGEST_MOD
    return int(total)


def _local_metadata(X, sharded: bool, with_checksum: bool):
    """Describe this rank's input as a fixed-width vector of integers."""
    if isinstance(X, DataLoader):
        dataset = getattr(X, "dataset", None)
        n_samples = len(dataset) if hasattr(dataset, "__len__") else _UNKNOWN
        # Streaming inputs are described only by what is already known; a
        # checksum would require an extra full pass over the data.
        from torchdr.distance.faiss import get_dataloader_metadata

        cached = get_dataloader_metadata(X)
        n_features = cached["n_features"] if cached else _UNKNOWN
        dtype_code = _dtype_code(cached["dtype"]) if cached else _UNKNOWN
        kind = _KIND_DATALOADER
        checksum = _UNKNOWN
    else:
        n_samples = int(X.shape[0])
        n_features = int(X.shape[1]) if X.dim() > 1 else _UNKNOWN
        dtype_code = _dtype_code(X.dtype)
        kind = _KIND_TENSOR
        checksum = _tensor_checksum(X) if with_checksum else _UNKNOWN

    return [n_samples, n_features, dtype_code, kind, checksum, int(sharded)]


def _collective_device(dist_ctx) -> torch.device:
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        return torch.device(f"cuda:{dist_ctx.local_rank}")
    return torch.device("cpu")


def _describe(field: str, value: int) -> str:
    if value == _UNKNOWN:
        return "unknown"
    if field == "dtype":
        return f"dtype-code {value}"
    if field == "input_kind":
        return "DataLoader" if value == _KIND_DATALOADER else "tensor"
    return str(value)


def validate_distributed_input(X, dist_ctx, verify_content: bool = True) -> None:
    """Check that every rank was given the same full dataset.

    Parameters
    ----------
    X : torch.Tensor or torch.utils.data.DataLoader
        This rank's input.
    dist_ctx : DistributedContext
        Active distributed context.
    verify_content : bool, default=True
        Also compare an order-sensitive checksum of the data across ranks.
        Only applies to tensor inputs; DataLoader inputs would need an extra
        full pass over the data.

    Raises
    ------
    ValueError
        If a rank's DataLoader is sharded, or if the ranks disagree on the
        shape, dtype, input type, or content of ``X``. The error is raised on
        every rank, not only on the offending one.
    """
    if dist_ctx is None or not dist_ctx.is_initialized:
        return

    shard_info = _loader_shard_info(X)

    world_size = dist_ctx.world_size
    if world_size < 2 or not dist.is_initialized():
        if shard_info is not None:
            raise ValueError(_sharded_loader_message(shard_info, ()))
        return

    device = _collective_device(dist_ctx)
    local = torch.tensor(
        _local_metadata(X, shard_info is not None, verify_content),
        dtype=torch.int64,
        device=device,
    )
    gathered = torch.empty(world_size * _VECTOR_LEN, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local)
    gathered = gathered.view(world_size, _VECTOR_LEN).cpu()

    sharded_ranks = torch.nonzero(gathered[:, _SHARDED_IDX] == 1).flatten().tolist()
    if sharded_ranks:
        raise ValueError(_sharded_loader_message(shard_info, sharded_ranks))

    for field_idx, field in enumerate(_FIELDS):
        column = gathered[:, field_idx]

        # A rank reporting _UNKNOWN could not measure the field; that is not
        # evidence of disagreement, so compare only the ranks that know it.
        known = torch.nonzero(column != _UNKNOWN).flatten()
        if known.numel() < 2:
            continue
        ref = int(known[0])
        expected = int(column[ref])
        mismatched = known[column[known] != expected]
        if mismatched.numel() == 0:
            continue

        bad = int(mismatched[0])
        detail = (
            f"rank {ref} reports {_describe(field, expected)} but rank {bad} "
            f"reports {_describe(field, int(column[bad]))}"
        )
        if field == "data_checksum":
            raise ValueError(
                "[TorchDR] Distributed mode requires every rank to hold the "
                "identical full dataset. The ranks agree on shape and dtype "
                f"but not on content: {detail} (order-sensitive checksum). "
                + _CONTRACT
                + " Check for per-rank shuffling, non-deterministic file "
                "ordering, or a sharded data pipeline."
            )
        raise ValueError(
            "[TorchDR] Distributed mode requires every rank to hold the "
            f"identical full dataset, but the ranks disagree on {field}: "
            f"{detail}. " + _CONTRACT
        )
