"""High-level execution plans for FAISS-backed distance computation."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .faiss import FaissConfig

_VALID_MODES = ("exact", "balanced", "fast")
_VALID_DISTRIBUTIONS = ("auto", "replicate", "shard")


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
        ``"auto"`` currently preserves the existing replicated-index strategy;
        automatic memory-aware selection is not yet implemented.
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


def _resolve_faiss_plan(
    config: FaissPlanConfig,
    *,
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
    distributed_ctx=None,
) -> Tuple[_FaissPlan, FaissConfig]:
    """Resolve user intent into diagnostics and a fresh low-level config."""
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
    elif config.distribution == "shard":
        if resolved.index_type != "Flat":
            raise NotImplementedError(
                "[TorchDR] distribution='shard' currently supports only exact "
                "Flat indexes. Use distribution='replicate' for an expert "
                "approximate index."
            )
        distribution = "shard"
    else:
        # Automatic memory-aware selection needs a trustworthy total peak-memory
        # estimate. Until that exists, preserve the established fast path.
        distribution = "replicate"

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
