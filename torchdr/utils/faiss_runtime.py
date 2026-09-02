"""Thread-local ownership of FAISS GPU resources."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import contextlib
import threading
from typing import Any, Dict, Optional, Tuple, Union

import torch

from torchdr.utils.faiss import faiss

# One FAISS resource per CUDA device and calling CPU thread. FAISS documents
# ``StandardGpuResources`` as not thread-safe because its temporary memory can
# only have one user at a time. A thread may, and should, share its resource
# between the GPU indexes it creates on the same device.
# ``StandardGpuResources`` holds a temporary memory pool, a pinned host buffer
# and per-device CUDA state, so building one per search costs both time and
# device memory. This runtime state is never user configuration: it is not
# copied, pickled, or reachable from an estimator's parameters. Thread-local
# ownership also releases a worker's resources when that thread exits.
_STATE = threading.local()


def _thread_state() -> Tuple[Dict[int, Any], Dict[int, Union[str, float]]]:
    """Return the resource and temp-memory maps for the calling thread."""
    if not hasattr(_STATE, "resources"):
        _STATE.resources = {}
        _STATE.temp_memory = {}
    return _STATE.resources, _STATE.temp_memory


def faiss_gpu_available() -> bool:
    """Whether the installed FAISS build can create GPU indexes."""
    return bool(faiss) and hasattr(faiss, "StandardGpuResources")


def get_gpu_resources(device_id: int, temp_memory: Union[str, float] = "auto") -> Any:
    """Return this thread's FAISS GPU resource for ``device_id``.

    Parameters
    ----------
    device_id : int
        CUDA device ordinal the resource serves.
    temp_memory : str or float, default='auto'
        Size in GB of the FAISS temporary memory pool, or ``'auto'`` to keep
        FAISS' default pool. An explicit size is applied only when it differs
        from the size in force for that device. Returning to ``'auto'`` after an
        explicit size recreates the resource because FAISS exposes no method to
        restore its version- and device-dependent default.

    Returns
    -------
    resources : faiss.StandardGpuResources
        The resource owned by the calling thread for ``device_id``.
    """
    if not faiss_gpu_available():
        raise RuntimeError(
            "[TorchDR] The installed FAISS build has no GPU support. "
            "Install `faiss-gpu` to run FAISS on CUDA devices."
        )

    device_id = int(device_id)

    resources, temp_memory_by_device = _thread_state()
    res = resources.get(device_id)
    current_temp_memory = temp_memory_by_device.get(device_id)

    if res is None or (temp_memory == "auto" and current_temp_memory != "auto"):
        res = faiss.StandardGpuResources()
        resources[device_id] = res
        temp_memory_by_device[device_id] = "auto"

    if temp_memory != "auto" and temp_memory != temp_memory_by_device[device_id]:
        res.setTempMemory(int(float(temp_memory) * 1024**3))
        temp_memory_by_device[device_id] = temp_memory

    return res


def reset_gpu_resources() -> None:
    """Release FAISS GPU resources held by the calling thread.

    Frees the temporary memory pools without waiting for interpreter shutdown.
    Intended for tests, which must not leak resources across cases.
    """
    resources, temp_memory_by_device = _thread_state()
    resources.clear()
    temp_memory_by_device.clear()


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
