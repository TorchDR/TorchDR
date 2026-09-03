"""Execution-plan API for FAISS-backed distance computation."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from .faiss import FaissConfig

_VALID_MODES = ("exact", "balanced", "fast")
_VALID_DISTRIBUTIONS = ("auto", "replicate", "shard")


class FaissPlanConfig:
    """High-level FAISS execution intent.

    Exposes accuracy and resource *intent* instead of the many low-level FAISS
    knobs. Exact search is the default; approximation and reduced precision are
    explicit opt-ins. Automatic values are resolved into an immutable execution
    plan (see :func:`resolve_faiss_plan`) that can be inspected after fitting
    through an estimator's or affinity's ``faiss_plan_`` attribute. This object
    is never mutated during resolution.

    Pass it wherever a ``backend`` is accepted, e.g.
    ``TSNE(backend=FaissPlanConfig(mode="exact"))``.

    Parameters
    ----------
    mode : {"exact", "balanced", "fast"}, default="exact"
        Accuracy/speed intent. ``"exact"`` uses a full-precision brute-force
        (Flat) index and never silently approximates. ``"balanced"`` and
        ``"fast"`` are explicit approximate opt-ins whose thresholds are not yet
        benchmarked (see issue #304); they raise :class:`NotImplementedError` at
        resolution time. For approximate search today, pass an ``expert``
        :class:`~torchdr.distance.FaissConfig`.
    distribution : {"auto", "replicate", "shard"}, default="auto"
        Multi-GPU execution topology. ``"auto"`` and ``"replicate"`` build a
        complete index on every rank (the current behavior) and do not change
        the global k-NN result. ``"shard"`` is not yet supported (see issue
        #301) and raises at resolution time.
    memory_budget : "auto" or int, default="auto"
        Target memory budget in bytes. Only ``"auto"`` is honored today; an
        explicit budget is not yet supported (see issue #301).
    random_state : int, optional
        Seed for deterministic sampling/training, reusing the estimator seed.
        The resolved plan records it.
    expert : FaissConfig, optional
        Expert override exposing supported low-level FAISS options. When set,
        ``mode`` must be left at its default ``"exact"`` and the given
        configuration is used verbatim as the resolved backend. Device ownership
        still comes from the distributed context.

    Examples
    --------
    >>> from torchdr.distance import FaissConfig, FaissPlanConfig

    >>> # Normal user: exact, full-precision k-NN (the default).
    >>> plan = FaissPlanConfig()

    >>> # Preset intent (approximate presets are not yet available; see #304).
    >>> plan = FaissPlanConfig(mode="exact", distribution="auto")

    >>> # Expert: full control over the low-level FAISS index.
    >>> plan = FaissPlanConfig(expert=FaissConfig(index_type="IVFPQ", nlist=1024))
    """

    def __init__(
        self,
        mode: str = "exact",
        distribution: str = "auto",
        memory_budget: Union[str, int] = "auto",
        random_state: Optional[int] = None,
        expert: Optional[FaissConfig] = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"[TorchDR] Unknown mode {mode!r}; expected one of {_VALID_MODES}."
            )
        if distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"[TorchDR] Unknown distribution {distribution!r}; "
                f"expected one of {_VALID_DISTRIBUTIONS}."
            )
        if not (
            memory_budget == "auto"
            or (
                isinstance(memory_budget, int)
                and not isinstance(memory_budget, bool)
                and memory_budget > 0
            )
        ):
            raise ValueError(
                "[TorchDR] memory_budget must be 'auto' or a positive integer "
                f"number of bytes; got {memory_budget!r}."
            )
        if expert is not None and not isinstance(expert, FaissConfig):
            raise TypeError(
                "[TorchDR] expert must be a FaissConfig or None; "
                f"got {type(expert).__name__}."
            )
        if expert is not None and mode != "exact":
            raise ValueError(
                "[TorchDR] Cannot combine an expert override with a non-default "
                "mode; pass either mode= (a preset) or expert= (a low-level "
                "FaissConfig), not both."
            )
        self.mode = mode
        self.distribution = distribution
        self.memory_budget = memory_budget
        self.random_state = random_state
        self.expert = expert

    def __repr__(self):
        parts = [f"mode={self.mode!r}", f"distribution={self.distribution!r}"]
        if self.memory_budget != "auto":
            parts.append(f"memory_budget={self.memory_budget!r}")
        if self.random_state is not None:
            parts.append(f"random_state={self.random_state!r}")
        if self.expert is not None:
            parts.append(f"expert={self.expert!r}")
        return f"FaissPlanConfig({', '.join(parts)})"


@dataclass(frozen=True, repr=False)
class _FaissPlan:
    """Immutable resolved FAISS execution plan.

    Recorded by an estimator/affinity as ``faiss_plan_`` and printed on rank 0
    when ``verbose=True``. Purely diagnostic: the low-level configuration that
    actually runs is returned separately by :func:`resolve_faiss_plan`.

    Attributes
    ----------
    index_type : str
        Resolved FAISS index, e.g. ``"Flat"``.
    precision : {"float32", "reduced"}
        ``"reduced"`` for product-quantized indexes, ``"float32"`` otherwise.
    distribution : {"replicate"}
        Resolved multi-GPU topology.
    training_size : int or None
        Number of rows used to train the index (``0`` for an untrained Flat
        index, ``None`` when resolved downstream for approximate indexes).
    batch_size : "auto" or int
        Rows handed to FAISS per add/search call.
    memory_estimate : int or None
        Estimated index memory in bytes, or ``None`` when it cannot be
        estimated upfront.
    random_state : int or None
        Seed recorded for deterministic sampling/training.
    """

    index_type: str
    precision: str
    distribution: str
    training_size: Optional[int]
    batch_size: Union[str, int]
    memory_estimate: Optional[int]
    random_state: Optional[int]

    def __repr__(self):
        mem = (
            "unknown"
            if self.memory_estimate is None
            else f"{self.memory_estimate} bytes"
        )
        train = "unknown" if self.training_size is None else self.training_size
        return (
            "FaissPlan("
            f"index_type={self.index_type!r}, precision={self.precision!r}, "
            f"distribution={self.distribution!r}, training_size={train}, "
            f"batch_size={self.batch_size!r}, memory_estimate={mem}, "
            f"random_state={self.random_state!r})"
        )


def _precision_of(index_type: str) -> str:
    """Precision label: product quantization compresses, everything else is full."""
    return "reduced" if "PQ" in index_type else "float32"


def resolve_faiss_plan(
    config: FaissPlanConfig,
    *,
    n_samples: Optional[int] = None,
    dim: Optional[int] = None,
    dist_ctx=None,
    device=None,
) -> Tuple[_FaissPlan, FaissConfig]:
    """Resolve a :class:`FaissPlanConfig` into a plan and a low-level FaissConfig.

    Pure and non-mutating: returns a new immutable :class:`_FaissPlan` and a new
    :class:`~torchdr.distance.FaissConfig`; the input ``config`` (and any
    ``expert`` it carries) are left untouched. Raises
    :class:`NotImplementedError` for intent that cannot yet be honored honestly.

    Parameters
    ----------
    config : FaissPlanConfig
        High-level execution intent.
    n_samples, dim : int, optional
        Dataset shape, used only to estimate index memory when available.
    dist_ctx : DistributedContext, optional
        Present for interface symmetry; device ownership in distributed mode is
        applied later via ``DistributedContext.get_faiss_config``.
    device : optional
        Present for interface symmetry; unused for the exact plan.

    Returns
    -------
    plan : _FaissPlan
        Immutable diagnostic plan (assign to ``faiss_plan_``).
    resolved : FaissConfig
        Low-level configuration to hand to ``pairwise_distances``.
    """
    # Distribution intent (independent of mode/expert).
    if config.distribution == "shard":
        raise NotImplementedError(
            "[TorchDR] distribution='shard' is not yet supported; see issue #301. "
            "Use 'auto' or 'replicate'."
        )
    resolved_distribution = "replicate"

    # Memory budget.
    if config.memory_budget != "auto":
        raise NotImplementedError(
            "[TorchDR] An explicit memory_budget is not yet supported; see issue "
            "#301. Use memory_budget='auto'."
        )

    # Resolve the low-level FAISS configuration.
    if config.expert is not None:
        # Expert override: copy the given config field-by-field so the user's
        # object is never mutated by downstream device handling.
        src = config.expert
        resolved = FaissConfig(
            temp_memory=src.temp_memory,
            device=src.device,
            index_type=src.index_type,
            nprobe=src.nprobe,
            nlist=src.nlist,
            M=src.M,
            nbits=src.nbits,
            stream_batch_size=src.stream_batch_size,
            **src.faiss_kwargs,
        )
    elif config.mode == "exact":
        resolved = FaissConfig(index_type="Flat")
    else:
        raise NotImplementedError(
            f"[TorchDR] mode={config.mode!r} is not yet supported; see issue #304. "
            "Only mode='exact' is available today. For approximate search, pass an "
            "expert FaissConfig, e.g. expert=FaissConfig(index_type='IVFPQ')."
        )

    index_type = resolved.index_type
    # A Flat index is untrained; approximate indexes train downstream on a
    # bounded sample whose size this plan does not fix.
    training_size = 0 if index_type == "Flat" else None
    if index_type == "Flat" and n_samples is not None and dim is not None:
        memory_estimate = int(n_samples) * int(dim) * 4  # float32 storage
    else:
        memory_estimate = None

    plan = _FaissPlan(
        index_type=index_type,
        precision=_precision_of(index_type),
        distribution=resolved_distribution,
        training_size=training_size,
        batch_size=resolved.stream_batch_size,
        memory_estimate=memory_estimate,
        random_state=config.random_state,
    )
    return plan, resolved
