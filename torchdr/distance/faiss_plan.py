"""High-level execution plans for FAISS-backed distance computation."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .faiss import FaissConfig

_VALID_MODES = ("exact", "balanced", "fast")
_VALID_DISTRIBUTIONS = ("auto", "replicate", "shard")

# Empirical per-GPU peak-memory model for exact ``Flat`` self k-NN, calibrated
# on the issue #301 memory-scaling sweep (single node, B200, FAISS 1.11.0).
# Under TorchDR's distributed-input contract the input ``X`` is replicated on
# every rank regardless of topology, so the resident footprint is: the
# replicated input, plus a ``Flat`` index whose per-row storage matches the
# input's, plus a fixed FAISS temporary-pool/context overhead. The calibrated
# boundary (replicate OOMs while shard fits at the predicted crossover) held to
# within a few percent, which the safety margin below absorbs along with the
# unmodeled k-NN output arrays and framework scratch.
_INPUT_BYTES_PER_ROW_PER_DIM = 4  # float32 input and Flat index storage
_FAISS_FIXED_OVERHEAD_BYTES = int(0.66 * 1024**3)  # ~0.66 GiB pool + context
_AUTO_MEMORY_SAFETY = 0.90  # fraction of per-GPU memory 'auto' will commit to


@dataclass(frozen=True)
class FaissPlanConfig:
    """Describe high-level intent for a FAISS execution.

    This is the small, user-facing alternative to :class:`FaissConfig`. The
    default always resolves to exact, full-precision Flat search. Approximate
    presets are represented now so their eventual behavior has a stable home,
    but fail explicitly until their benchmarked implementations land.

    Parameters
    ----------
    mode : {"exact", "balanced", "fast"}, default="exact"
        Accuracy/speed intent. Only ``"exact"`` is currently implemented;
        ``"balanced"`` and ``"fast"`` raise :class:`NotImplementedError`.
    distribution : {"auto", "replicate", "shard"}, default="auto"
        Multi-GPU topology. ``"replicate"`` builds the full index on every rank.
        ``"shard"`` splits an exact ``Flat`` index across ranks. The input tensor
        remains replicated under TorchDR's current distributed-input contract.
        ``"auto"`` keeps the replicated fast path when a full index fits the
        measured per-GPU memory, shards an exact ``Flat`` index when it does not,
        and raises rather than launch a run estimated to run out of memory. It
        falls back to ``"replicate"`` for an expert (non-Flat) index, which
        cannot be sharded.
    expert : FaissConfig, optional
        Explicit low-level override for advanced users. It cannot be combined
        with a non-default mode. The object is copied during resolution.

    Examples
    --------
    >>> from torchdr.distance import FaissConfig, FaissPlanConfig
    >>> exact = FaissPlanConfig()
    >>> replicated = FaissPlanConfig(distribution="replicate")
    >>> expert = FaissPlanConfig(expert=FaissConfig(index_type="IVF", nprobe=8))
    """

    mode: str = "exact"
    distribution: str = "auto"
    expert: Optional[FaissConfig] = None

    def __post_init__(self):
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"[TorchDR] Unknown mode {self.mode!r}; expected one of {_VALID_MODES}."
            )
        if self.distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"[TorchDR] Unknown distribution {self.distribution!r}; "
                f"expected one of {_VALID_DISTRIBUTIONS}."
            )
        if self.expert is not None and not isinstance(self.expert, FaissConfig):
            raise TypeError(
                "[TorchDR] expert must be a FaissConfig or None; "
                f"got {type(self.expert).__name__}."
            )
        if self.expert is not None and self.mode != "exact":
            raise ValueError(
                "[TorchDR] Cannot combine an expert override with a non-default "
                "mode. Pass either mode= or expert=."
            )


@dataclass(frozen=True, repr=False)
class _FaissPlan:
    """Resolved, immutable diagnostics exposed as ``faiss_plan_``.

    ``index_memory_bytes`` estimates only FAISS's stored database vectors on
    each rank. It deliberately excludes inputs, outputs, and temporary scratch
    rather than presenting a partial estimate as total peak memory.
    """

    mode: str
    index_type: str
    precision: str
    distribution: str
    training_size: Optional[int]
    stream_batch_size: Union[str, int]
    index_memory_bytes: Optional[int]

    def __repr__(self):
        return (
            f"FaissPlan(mode={self.mode!r}, index_type={self.index_type!r}, "
            f"precision={self.precision!r}, distribution={self.distribution!r}, "
            f"training_size={self.training_size!r}, "
            f"stream_batch_size={self.stream_batch_size!r}, "
            f"index_memory_bytes={self.index_memory_bytes!r})"
        )


def _copy_config(config: FaissConfig) -> FaissConfig:
    """Copy a low-level config without retaining its mutable kwargs mapping."""
    return FaissConfig(
        temp_memory=config.temp_memory,
        device=config.device,
        index_type=config.index_type,
        nprobe=config.nprobe,
        nlist=config.nlist,
        M=config.M,
        nbits=config.nbits,
        stream_batch_size=config.stream_batch_size,
        **config.faiss_kwargs,
    )


def _estimate_search_peak_bytes(
    n_samples: int, n_features: int, index_rows: int
) -> int:
    """Estimate peak per-GPU bytes for exact ``Flat`` self k-NN.

    See the calibrated model documented alongside the module constants: a fixed
    FAISS overhead, plus the replicated input, plus ``index_rows`` of Flat index
    storage. ``index_rows`` equals ``n_samples`` when the index is replicated and
    the per-rank shard size when it is sharded.
    """
    row_bytes = _INPUT_BYTES_PER_ROW_PER_DIM * int(n_features)
    input_bytes = row_bytes * int(n_samples)  # replicated on every rank
    index_bytes = row_bytes * int(index_rows)  # Flat index storage
    return _FAISS_FIXED_OVERHEAD_BYTES + input_bytes + index_bytes


def _select_auto_distribution(
    *,
    n_samples: Optional[int],
    n_features: Optional[int],
    world_size: Optional[int],
    available_memory_bytes: Optional[int],
    safety_margin: float = _AUTO_MEMORY_SAFETY,
) -> str:
    """Choose ``replicate`` or ``shard`` under a per-GPU memory budget.

    Returns ``"replicate"`` when a full index fits per rank (the faster path with
    no per-query collectives), ``"shard"`` when only a sharded index fits, and
    raises :class:`RuntimeError` rather than launch a run estimated to run out of
    memory. When the budget or the input size is unknown, or there is a single
    rank, it preserves the replicated fast path so callers that cannot measure
    memory (e.g. the diagnostic ``faiss_plan_`` pass) keep the established
    behavior.

    The selection is a pure function of its arguments so every rank that is given
    the same budget reaches the same decision; the caller is responsible for
    reducing the budget to a single cross-rank value before calling.
    """
    if (
        available_memory_bytes is None
        or n_samples is None
        or n_features is None
        or world_size is None
        or int(world_size) <= 1
    ):
        return "replicate"

    budget = safety_margin * float(available_memory_bytes)
    replicate_peak = _estimate_search_peak_bytes(n_samples, n_features, n_samples)
    if replicate_peak <= budget:
        return "replicate"

    world_size = int(world_size)
    shard_rows = (int(n_samples) + world_size - 1) // world_size
    shard_peak = _estimate_search_peak_bytes(n_samples, n_features, shard_rows)
    if shard_peak <= budget:
        return "shard"

    raise RuntimeError(
        "[TorchDR] distribution='auto' estimates ~"
        f"{replicate_peak / 1e9:.1f} GB per GPU for a replicated index and ~"
        f"{shard_peak / 1e9:.1f} GB even when sharded across {world_size} ranks, "
        f"but only ~{budget / 1e9:.1f} GB is safely usable per GPU. Reduce the "
        "dataset, add ranks, or use an approximate expert index. 'auto' refuses "
        "rather than start a run that is estimated to run out of memory."
    )


def _resolve_faiss_plan(
    config: FaissPlanConfig,
    *,
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
    distributed_ctx=None,
    available_memory_bytes: Optional[int] = None,
) -> Tuple[_FaissPlan, FaissConfig]:
    """Resolve user intent into diagnostics and a fresh low-level config.

    ``available_memory_bytes`` is the single cross-rank per-GPU budget used to
    resolve ``distribution='auto'``. It is ``None`` on the diagnostic path (no
    measurement), which keeps ``auto`` on the replicated fast path; the
    distributed dispatcher passes a measured, reduced value so ``auto`` can
    select sharding or refuse an over-budget run.
    """
    if config.expert is not None:
        resolved = _copy_config(config.expert)
        mode = "expert"
    elif config.mode == "exact":
        resolved = FaissConfig(index_type="Flat")
        mode = "exact"
    else:
        raise NotImplementedError(
            f"[TorchDR] mode={config.mode!r} is not yet supported; see issue #304."
        )

    is_distributed = bool(
        distributed_ctx is not None and distributed_ctx.is_initialized
    )
    world_size = int(getattr(distributed_ctx, "world_size", 1)) if is_distributed else 1
    training_size = 0 if resolved.index_type == "Flat" else None
    if not is_distributed or world_size <= 1:
        distribution = "single"
    elif config.distribution == "replicate":
        distribution = "replicate"
    elif config.distribution == "shard":
        if resolved.index_type != "Flat":
            raise NotImplementedError(
                "[TorchDR] distribution='shard' currently supports only exact "
                "Flat indexes. Use distribution='replicate' for an expert "
                "approximate index."
            )
        distribution = "shard"
    elif resolved.index_type != "Flat":
        # 'auto' can only shard exact Flat indexes; an expert approximate index
        # stays replicated because sharding it is not implemented.
        distribution = "replicate"
    else:
        # 'auto': pick the fastest topology that fits the measured per-GPU
        # budget, or refuse. Without a budget (diagnostic pass) this is a no-op
        # that keeps the established replicated fast path.
        distribution = _select_auto_distribution(
            n_samples=n_samples,
            n_features=n_features,
            world_size=world_size,
            available_memory_bytes=available_memory_bytes,
        )

    index_memory_bytes = None
    if training_size == 0 and n_samples is not None and n_features is not None:
        indexed_rows = int(n_samples)
        if distribution == "shard":
            indexed_rows = (indexed_rows + world_size - 1) // world_size
        index_memory_bytes = indexed_rows * int(n_features) * 4

    return (
        _FaissPlan(
            mode=mode,
            index_type=resolved.index_type,
            precision="reduced" if resolved.index_type == "IVFPQ" else "float32",
            distribution=distribution,
            training_size=training_size,
            stream_batch_size=resolved.stream_batch_size,
            index_memory_bytes=index_memory_bytes,
        ),
        resolved,
    )
