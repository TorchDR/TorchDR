"""Process-level ownership of FAISS GPU resources."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import contextlib
import threading
from typing import Any, Dict, Optional, Union

import torch

from torchdr.utils.faiss import faiss

# One FAISS resource per CUDA device, owned by the process for its lifetime.
# ``StandardGpuResources`` holds a temporary memory pool, a pinned host buffer
# and per-device CUDA state, so building one per search costs both time and
# device memory. This is runtime state, never user configuration: it is not
# copied, pickled, or reachable from an estimator's parameters.
_LOCK = threading.Lock()
_RESOURCES: Dict[int, Any] = {}
_TEMP_MEMORY: Dict[int, Union[str, float]] = {}


def faiss_gpu_available() -> bool:
    """Whether the installed FAISS build can create GPU indexes."""
    return bool(faiss) and hasattr(faiss, "StandardGpuResources")


def get_gpu_resources(device_id: int, temp_memory: Union[str, float] = "auto") -> Any:
    """Return this process' FAISS GPU resource for ``device_id``.

    Parameters
    ----------
    device_id : int
        CUDA device ordinal the resource serves.
    temp_memory : str or float, default='auto'
        Size in GB of the FAISS temporary memory pool, or ``'auto'`` to keep
        whatever pool FAISS already sized. An explicit size is applied only when
        it differs from the size in force for that device. ``'auto'`` never
        resizes an existing pool: FAISS exposes no call that restores its own
        default, so honouring it would shrink a pool an earlier caller asked for.

    Returns
    -------
    resources : faiss.StandardGpuResources
        The resource owned by this process for ``device_id``.
    """
    if not faiss_gpu_available():
        raise RuntimeError(
            "[TorchDR] The installed FAISS build has no GPU support. "
            "Install `faiss-gpu` to run FAISS on CUDA devices."
        )

    device_id = int(device_id)

    with _LOCK:
        res = _RESOURCES.get(device_id)
        if res is None:
            res = faiss.StandardGpuResources()
            _RESOURCES[device_id] = res
            _TEMP_MEMORY[device_id] = "auto"

        if temp_memory != "auto" and temp_memory != _TEMP_MEMORY[device_id]:
            res.setTempMemory(int(float(temp_memory) * 1024**3))
            _TEMP_MEMORY[device_id] = temp_memory

        return res


def reset_gpu_resources() -> None:
    """Release every FAISS GPU resource held by this process.

    Frees the temporary memory pools without waiting for interpreter shutdown.
    Intended for tests, which must not leak resources across cases.
    """
    with _LOCK:
        _RESOURCES.clear()
        _TEMP_MEMORY.clear()


@contextlib.contextmanager
def faiss_device_scope(device: Optional[torch.device]):
    """Make ``device`` current so FAISS and PyTorch agree on the CUDA stream.

    ``faiss.contrib.torch_utils`` wraps every ``train``/``add``/``search`` call
    in ``using_stream``, which binds the resource to ``torch.cuda.current_stream()``
    of ``torch.cuda.current_device()``. That binding is what orders FAISS behind
    the writes PyTorch made to the input tensors, but it targets the current
    device, so an index living on another device keeps its own FAISS stream and
    can read writes that have not completed. Entering the index device first
    keeps both libraries on one stream, which is the only synchronization the
    tensor interoperability path needs.

    Parameters
    ----------
    device : torch.device or None
        Device the FAISS index lives on. ``None`` and CPU devices are no-ops.
    """
    if device is None or device.type != "cuda":
        yield
        return

    with torch.cuda.device(device):
        yield
