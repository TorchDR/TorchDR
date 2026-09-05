"""Validation for inputs used by distributed nearest-neighbor search."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

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
    "TorchDR's distributed nearest-neighbor search currently requires the same "
    "full dataset on every rank."
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


def collective_device(dist_ctx) -> torch.device:
    """Device a collective's tensors must live on for the active backend."""
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

    device = collective_device(dist_ctx)
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

    # Reject more ranks than samples. Only the query rows are partitioned, so
    # with world_size > n_samples some ranks own zero rows and the index->owner
    # lookup divides by a zero-sized block (ZeroDivisionError deep in sparse
    # symmetrization). Catch it here with an actionable message.
    n_samples_column = gathered[:, _FIELDS.index("n_samples")]
    known = n_samples_column[n_samples_column != _UNKNOWN]
    if known.numel() and int(known.min()) < dist_ctx.world_size:
        raise ValueError(
            f"[TorchDR] distributed neighbor embedding needs at least one "
            f"sample per rank, but n_samples={int(known.min())} with "
            f"world_size={dist_ctx.world_size}. Launch with at most n_samples "
            f"ranks. {_CONTRACT}"
        )


# Metadata exchanged for an explicitly sharded input: (n_local, n_features,
# dtype-code). Unlike the replicated contract above, the row count is expected to
# differ between ranks, so it is summed into a global count rather than compared.
_SHARD_META_LEN = 3


@dataclass(frozen=True)
class ShardLayout:
    """Rank-major layout of an input whose rows are split across ranks.

    Each rank holds a distinct contiguous shard ``X_local`` of a global dataset;
    concatenating the shards in rank order reconstructs the full input. This
    describes where this rank's rows sit in that global ordering, which is what
    lets a distributed search return *global* neighbor indices and lets sparse
    affinity find the owner of an arbitrary global row.

    Attributes
    ----------
    rank : int
        Rank of the current process.
    world_size : int
        Number of ranks the input is split across.
    local_count : int
        Rows held by this rank (may be zero).
    global_count : int
        Total rows across all ranks.
    local_offset : int
        Global index of this rank's first row (rank-major prefix sum). Local row
        ``i`` is therefore global row ``local_offset + i``.
    counts : tuple of int
        Row count of every rank, in rank order. ``counts[r]`` and the running
        prefix give the half-open global range each rank owns.
    """

    rank: int
    world_size: int
    local_count: int
    global_count: int
    local_offset: int
    counts: Tuple[int, ...]

    def query_ids(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Global row indices of this rank's local rows."""
        return torch.arange(
            self.local_offset,
            self.local_offset + self.local_count,
            device=device,
        )

    def owner_boundaries(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Rank-major prefix offsets ``[0, c0, c0 + c1, ..., global_count]``.

        A length ``world_size + 1`` table where ``boundaries[r]`` is the global
        index of rank ``r``'s first row and ``boundaries[r + 1]`` is one past its
        last, so a global row ``g`` is owned by the rank ``r`` for which
        ``boundaries[r] <= g < boundaries[r + 1]``. This is the exact table
        :func:`torchdr.utils.sparse.distributed_symmetrize_sparse` consumes to
        route an arbitrary global column to its owner when the shards are uneven;
        with the balanced split it reduces to that function's default arithmetic.
        """
        boundaries = torch.zeros(self.world_size + 1, dtype=torch.long, device=device)
        boundaries[1:] = torch.as_tensor(
            self.counts, dtype=torch.long, device=device
        ).cumsum(0)
        return boundaries


def gather_shard_layout(X_local, dist_ctx) -> ShardLayout:
    """Exchange local row counts and derive a rank-major shard layout.

    Every rank passes only its own contiguous shard ``X_local``; this collects
    each rank's row count once with a single ``all_gather``, sums them into the
    global sample count, and derives this rank's rank-major prefix offset. The
    feature count and dtype are validated collectively so a mismatched shard
    fails loudly instead of silently returning wrong neighbors.

    Empty ranks (zero local rows) are supported at this layer: a ``(0, d)`` shard
    still reports ``d`` features, still takes part in the exchange, and receives a
    well-defined offset. Callers that cannot give meaning to an empty shard (an
    affinity that would divide by a zero-sized block) should reject it themselves.

    Parameters
    ----------
    X_local : torch.Tensor of shape (n_local, n_features)
        This rank's contiguous shard of the global input.
    dist_ctx : DistributedContext or None
        The group across which the input is sharded. When ``None``, not
        initialized, or single-rank, the shard is treated as the whole dataset
        and no collective is issued.

    Returns
    -------
    ShardLayout
        The rank-major layout described above.
    """
    if X_local.ndim != 2:
        raise ValueError(
            f"[TorchDR] a sharded input must be a 2-D (n_local, n_features) "
            f"tensor, got shape {tuple(X_local.shape)}."
        )
    local_count = int(X_local.shape[0])
    n_features = int(X_local.shape[1])

    single_process = (
        dist_ctx is None
        or not dist_ctx.is_initialized
        or dist_ctx.world_size < 2
        or not dist.is_initialized()
    )
    if single_process:
        return ShardLayout(
            rank=0,
            world_size=1,
            local_count=local_count,
            global_count=local_count,
            local_offset=0,
            counts=(local_count,),
        )

    device = collective_device(dist_ctx)
    local = torch.tensor(
        [local_count, n_features, _dtype_code(X_local.dtype)],
        dtype=torch.int64,
        device=device,
    )
    gathered = torch.empty(
        dist_ctx.world_size * _SHARD_META_LEN, dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(gathered, local)
    gathered = gathered.view(dist_ctx.world_size, _SHARD_META_LEN).cpu()

    counts = [int(c) for c in gathered[:, 0].tolist()]
    for column_idx, field, describe in (
        (1, "n_features", str),
        (2, "dtype", lambda v: f"dtype-code {v}"),
    ):
        column = gathered[:, column_idx]
        reference = int(column[0])
        mismatches = torch.nonzero(column != reference).flatten()
        if mismatches.numel():
            bad = int(mismatches[0])
            raise ValueError(
                f"[TorchDR] ranks disagree on {field}: rank 0 reports "
                f"{describe(reference)}, while rank {bad} reports "
                f"{describe(int(column[bad]))}. A sharded input must split rows "
                f"only; every rank must share the same features and dtype."
            )

    rank = dist_ctx.rank
    return ShardLayout(
        rank=rank,
        world_size=dist_ctx.world_size,
        local_count=local_count,
        global_count=int(sum(counts)),
        local_offset=int(sum(counts[:rank])),
        counts=tuple(counts),
    )
