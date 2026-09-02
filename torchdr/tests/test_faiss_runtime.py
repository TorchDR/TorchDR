"""Tests for the process-level FAISS GPU runtime and its stream contract."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import copy
import pickle

import pytest
import torch
from torch.testing import assert_close

from torchdr.distance import FaissConfig, pairwise_distances
from torchdr.distance.faiss import pairwise_distances_faiss
from torchdr.utils import faiss
from torchdr.utils.faiss_runtime import (
    faiss_device_scope,
    faiss_gpu_available,
    get_gpu_resources,
    reset_gpu_resources,
)

requires_faiss = pytest.mark.skipif(not faiss, reason="faiss is not available")
requires_faiss_gpu = pytest.mark.skipif(
    not faiss or not torch.cuda.is_available() or not faiss_gpu_available(),
    reason="requires CUDA and a faiss-gpu build",
)
requires_two_gpus = pytest.mark.skipif(
    not faiss
    or not faiss_gpu_available()
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices and a faiss-gpu build",
)


@pytest.fixture(autouse=True)
def clean_runtime():
    """No test may observe or leak another test's FAISS resources."""
    reset_gpu_resources()
    yield
    reset_gpu_resources()


class _FakeResources:
    """Stand-in for ``StandardGpuResources`` recording temp-memory requests."""

    def __init__(self):
        self.temp_memory_calls = []

    def setTempMemory(self, size):  # noqa: N802 - mirrors the FAISS spelling
        self.temp_memory_calls.append(size)


class _FakeFaiss:
    StandardGpuResources = _FakeResources


@pytest.fixture
def fake_faiss(monkeypatch):
    """Drive the runtime with a fake FAISS module, so it is testable on CPU."""
    monkeypatch.setattr("torchdr.utils.faiss_runtime.faiss", _FakeFaiss)


# ====== resource ownership ======


def test_missing_gpu_support_raises_an_actionable_error(monkeypatch):
    monkeypatch.setattr("torchdr.utils.faiss_runtime.faiss", object())

    assert not faiss_gpu_available()
    with pytest.raises(RuntimeError, match="no GPU support"):
        get_gpu_resources(0)


def test_one_resource_per_device_never_shared_across_devices(fake_faiss):
    assert get_gpu_resources(0) is get_gpu_resources(0)
    assert get_gpu_resources(0) is not get_gpu_resources(1)


def test_explicit_temp_memory_is_applied_once_and_auto_keeps_the_pool(fake_faiss):
    res = get_gpu_resources(0, temp_memory=2.0)
    get_gpu_resources(0, temp_memory=2.0)
    assert res.temp_memory_calls == [int(2.0 * 1024**3)]

    # 'auto' must not shrink a pool an earlier caller sized.
    get_gpu_resources(0, temp_memory="auto")
    assert res.temp_memory_calls == [int(2.0 * 1024**3)]

    get_gpu_resources(0, temp_memory=1.0)
    assert res.temp_memory_calls[-1] == 1024**3


def test_reset_releases_every_resource(fake_faiss):
    first = get_gpu_resources(0)
    reset_gpu_resources()

    assert get_gpu_resources(0) is not first


# ====== configuration holds no runtime state ======


@requires_faiss
def test_config_stays_plain_data_after_a_search():
    config = FaissConfig()
    X = torch.randn(64, 8)

    pairwise_distances(X, k=5, backend=config, device="cpu")

    assert not any(
        type(value).__module__.startswith("faiss") for value in vars(config).values()
    )
    assert repr(config) == repr(FaissConfig())
    for clone in (copy.deepcopy(config), pickle.loads(pickle.dumps(config))):
        assert vars(clone) == vars(config)


@requires_faiss_gpu
def test_config_holds_no_gpu_resources_after_a_gpu_search():
    config = FaissConfig()
    X = torch.randn(256, 16, device="cuda")

    pairwise_distances(X, k=5, backend=config)

    assert vars(config) == vars(FaissConfig())
    assert pickle.loads(pickle.dumps(config)).index_type == "Flat"


@requires_faiss_gpu
def test_repeated_searches_share_one_resource():
    X = torch.randn(512, 16, device="cuda")

    pairwise_distances(X, k=5, backend="faiss")
    resources = get_gpu_resources(0)
    pairwise_distances(X, k=5, backend=FaissConfig())

    assert get_gpu_resources(0) is resources


# ====== same-device tensors reach FAISS without a host copy ======


@requires_faiss_gpu
def test_gpu_search_makes_no_host_copy(monkeypatch):
    from torchdr.utils.faiss import faiss_torch_interop

    if not faiss_torch_interop:
        pytest.skip("installed FAISS does not provide PyTorch interoperability")

    host_copies = []
    original_cpu = torch.Tensor.cpu

    def counting_cpu(self, *args, **kwargs):
        host_copies.append(tuple(self.shape))
        return original_cpu(self, *args, **kwargs)

    X = torch.randn(512, 16, device="cuda")
    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    distances, indices = pairwise_distances_faiss(X, k=5)

    assert host_copies == []
    assert distances.device.type == "cuda"
    assert indices.device.type == "cuda"


# ====== stream and device scoping ======


@pytest.mark.parametrize("device", [None, torch.device("cpu")])
def test_device_scope_is_a_no_op_off_cuda(device):
    with faiss_device_scope(device):
        pass


@requires_two_gpus
def test_device_scope_enters_the_index_device_and_restores_the_previous_one():
    with torch.cuda.device(0):
        with faiss_device_scope(torch.device("cuda", 1)):
            assert torch.cuda.current_device() == 1
        assert torch.cuda.current_device() == 0


def _queued_work(base, rotation, n_steps=32):
    """Enough asynchronous work that a racing read would observe stale memory."""
    out = base
    for _ in range(n_steps):
        out = torch.tanh(out @ rotation)
    return out


@requires_faiss_gpu
def test_search_consumes_writes_issued_on_a_non_default_stream():
    """add/search must be ordered behind tensor writes on the caller's stream."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    base = torch.randn(20000, 64, device="cuda", generator=generator)
    rotation = torch.randn(64, 64, device="cuda", generator=generator) / 8.0

    torch.cuda.synchronize()
    reference = _queued_work(base, rotation)
    expected_distances, expected_indices = pairwise_distances_faiss(reference, k=10)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        X = _queued_work(base, rotation)
        distances, indices = pairwise_distances_faiss(X, k=10)
        # Consuming the outputs on the same stream must also be ordered.
        checksum = distances.sum()
    torch.cuda.synchronize()

    assert torch.isfinite(checksum)
    assert torch.equal(indices, expected_indices)
    assert_close(distances, expected_distances)


@requires_two_gpus
def test_search_on_a_secondary_device_uses_that_device_stream():
    """The index device drives the stream even when another device is current."""
    generator = torch.Generator(device="cuda:1").manual_seed(0)
    base = torch.randn(20000, 64, device="cuda:1", generator=generator)
    rotation = torch.randn(64, 64, device="cuda:1", generator=generator) / 8.0
    config = FaissConfig(device=1)

    torch.cuda.synchronize()
    reference = _queued_work(base, rotation)
    expected_distances, expected_indices = pairwise_distances_faiss(
        reference, k=10, config=config
    )
    torch.cuda.synchronize()

    stream = torch.cuda.Stream(device="cuda:1")
    with torch.cuda.device(0), torch.cuda.stream(stream):
        X = _queued_work(base, rotation)
        distances, indices = pairwise_distances_faiss(X, k=10, config=config)
    torch.cuda.synchronize()

    assert distances.device == torch.device("cuda", 1)
    assert torch.equal(indices, expected_indices)
    assert_close(distances, expected_distances)
