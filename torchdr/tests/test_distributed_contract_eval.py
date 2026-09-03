"""Tests for the distributed-usage contract of the evaluation metrics.

The distributed launch and device guards in ``neighborhood_preservation`` and
``knn_label_accuracy`` mirror the estimator guards covered in
``test_distributed_contract.py``. They are asserted here so a refactor cannot
silently drop the actionable error a user gets when ``distributed=True`` is
requested off a launcher or on a CPU device.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr.eval import knn_label_accuracy, neighborhood_preservation


def _neighborhood_preservation(**kwargs):
    X = torch.randn(20, 5)
    Z = torch.randn(20, 2)
    return neighborhood_preservation(X, Z, K=5, **kwargs)


def _knn_label_accuracy(**kwargs):
    X = torch.randn(20, 5)
    labels = torch.randint(0, 3, (20,))
    return knn_label_accuracy(X, labels, k=5, **kwargs)


# Distributed evaluation metrics keyed by a caller that supplies valid inputs so
# only the distributed contract, not input validation, is under test.
DISTRIBUTED_EVAL_METRICS = [
    pytest.param(_neighborhood_preservation, id="neighborhood_preservation"),
    pytest.param(_knn_label_accuracy, id="knn_label_accuracy"),
]


@pytest.fixture
def no_process_group(monkeypatch):
    """Guarantee no live group so the launch guard is exercised deterministically."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


@pytest.fixture
def initialized_cpu_group(monkeypatch):
    """Simulate a live single-rank group without touching a CUDA device."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.parametrize("metric", DISTRIBUTED_EVAL_METRICS)
def test_eval_metric_distributed_true_requires_torchrun(metric, no_process_group):
    """``distributed=True`` outside a process group raises, not proceeds."""
    with pytest.raises(RuntimeError, match="torchrun"):
        metric(distributed=True)


@pytest.mark.parametrize("metric", DISTRIBUTED_EVAL_METRICS)
def test_eval_metric_distributed_cpu_device_requires_gpu(metric, initialized_cpu_group):
    """A live group with ``device='cpu'`` is rejected with a GPU message."""
    with pytest.raises(ValueError, match="GPU"):
        metric(distributed=True, device="cpu")
