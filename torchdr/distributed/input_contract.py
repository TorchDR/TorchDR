"""Validation for inputs used by distributed nearest-neighbor search."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import zlib

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


_UNKNOWN = -1
_KIND_TENSOR = 0
_KIND_DATALOADER = 1
_FIELDS = ("n_samples", "n_features", "dtype", "input_kind")
_SHARDED_IDX = len(_FIELDS)
_VECTOR_LEN = _SHARDED_IDX + 1

_CONTRACT = (
    "TorchDR builds a complete nearest-neighbor index on every rank and "
    "partitions only the query rows. Pass the same full dataset to every rank."
)


def _dtype_code(dtype) -> int:
    """Return a process-independent integer for a dtype."""
    return int(zlib.crc32(str(dtype).encode()))


def _loader_shard_info(dataloader):
    """Describe a loader that yields fewer samples than its dataset."""
    if not isinstance(dataloader, DataLoader):
        return None

    dataset = getattr(dataloader, "dataset", None)
    sampler = getattr(dataloader, "sampler", None)
    if dataset is None or sampler is None:
        return None

    try:
        n_total = len(dataset)
        n_iterated = len(sampler)
    except TypeError:
        return None

    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if getattr(batch_sampler, "drop_last", False):
        batch_size = getattr(batch_sampler, "batch_size", None)
        if isinstance(batch_size, int) and batch_size > 0:
            n_iterated -= n_iterated % batch_size

    if n_iterated == n_total:
        return None
    return n_iterated, n_total, type(sampler).__name__


def _local_metadata(X, sharded: bool):
    """Encode input metadata in a fixed-width integer vector."""
    if isinstance(X, DataLoader):
        dataset = getattr(X, "dataset", None)
        try:
            n_samples = len(dataset)
        except TypeError:
            n_samples = _UNKNOWN
        n_features = _UNKNOWN
        dtype = _UNKNOWN
        kind = _KIND_DATALOADER
    else:
        n_samples = int(X.shape[0])
        n_features = int(X.shape[1]) if X.ndim > 1 else _UNKNOWN
        dtype = _dtype_code(X.dtype)
        kind = _KIND_TENSOR

    return [n_samples, n_features, dtype, kind, int(sharded)]


def _collective_device(dist_ctx) -> torch.device:
    if dist.get_backend() == "nccl":
        return torch.device("cuda", dist_ctx.local_rank)
    return torch.device("cpu")


def _describe(field: str, value: int) -> str:
    if field == "input_kind":
        return "DataLoader" if value == _KIND_DATALOADER else "tensor"
    if field == "dtype":
        return f"dtype-code {value}"
    return str(value)


def _sharded_loader_error(info, ranks) -> ValueError:
    if info is None:
        detail = f"a DataLoader is sharded on rank(s) {', '.join(map(str, ranks))}"
    else:
        n_iterated, n_total, sampler_name = info
        detail = (
            f"{sampler_name} yields {n_iterated} of this rank's "
            f"{n_total} dataset samples"
        )
    return ValueError(f"[TorchDR] {detail}. {_CONTRACT}")


def validate_distributed_input(X, dist_ctx) -> None:
    """Reject detectable violations of the replicated-input contract.

    The check catches sharded DataLoaders and rank disagreements in input kind,
    shape, or dtype. It deliberately does not hash tensor contents: an exact
    comparison would add work proportional to the full dataset, while a sampled
    checksum could only provide a probabilistic guarantee.
    """
    if dist_ctx is None or not dist_ctx.is_initialized:
        return

    shard_info = _loader_shard_info(X)
    if dist_ctx.world_size < 2 or not dist.is_initialized():
        if shard_info is not None:
            raise _sharded_loader_error(shard_info, ())
        return

    device = _collective_device(dist_ctx)
    local = torch.tensor(
        _local_metadata(X, shard_info is not None), dtype=torch.int64, device=device
    )
    gathered = torch.empty(
        dist_ctx.world_size * _VECTOR_LEN, dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(gathered, local)
    gathered = gathered.view(dist_ctx.world_size, _VECTOR_LEN).cpu()

    sharded_ranks = torch.nonzero(gathered[:, _SHARDED_IDX]).flatten().tolist()
    if sharded_ranks:
        raise _sharded_loader_error(shard_info, sharded_ranks)

    for field_idx, field in enumerate(_FIELDS):
        column = gathered[:, field_idx]
        known_ranks = torch.nonzero(column != _UNKNOWN).flatten()
        if known_ranks.numel() < 2:
            continue

        reference_rank = int(known_ranks[0])
        reference = int(column[reference_rank])
        mismatches = known_ranks[column[known_ranks] != reference]
        if mismatches.numel() == 0:
            continue

        mismatch = int(mismatches[0])
        raise ValueError(
            f"[TorchDR] ranks disagree on {field}: rank {reference_rank} reports "
            f"{_describe(field, reference)}, while rank {mismatch} reports "
            f"{_describe(field, int(column[mismatch]))}. {_CONTRACT}"
        )
