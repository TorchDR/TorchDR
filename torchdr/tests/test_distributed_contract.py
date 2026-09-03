"""Tests for the public distributed-usage contract."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr.affinity import UMAPAffinity
from torchdr.eval import knn_label_accuracy, neighborhood_preservation
from torchdr.neighbor_embedding import COSNE, SNE, TSNE, UMAP, InfoTSNE, LargeVis

# Estimators that support distributed mode. PACMAP and TSNEkhorn are excluded:
# they reject distributed unconditionally and are covered elsewhere.
DISTRIBUTED_ESTIMATORS = [UMAP, InfoTSNE, LargeVis, SNE, TSNE, COSNE]


def _neighborhood_preservation(**kwargs):
    return neighborhood_preservation(
        torch.zeros(20, 5), torch.zeros(20, 2), K=5, **kwargs
    )


def _knn_label_accuracy(**kwargs):
    return knn_label_accuracy(torch.zeros(20, 5), torch.arange(20) % 3, k=5, **kwargs)


DISTRIBUTED_EVAL_METRICS = [
    pytest.param(_neighborhood_preservation, id="neighborhood_preservation"),
    pytest.param(_knn_label_accuracy, id="knn_label_accuracy"),
]


@pytest.fixture
def no_process_group(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


@pytest.fixture
def initialized_cpu_group(monkeypatch):
    """Simulate a live single-rank group without touching a CUDA device."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.parametrize("estimator", DISTRIBUTED_ESTIMATORS)
def test_estimator_distributed_true_requires_torchrun(estimator):
    """``distributed=True`` outside a process group raises, not proceeds."""
    with pytest.raises(RuntimeError, match="torchrun"):
        estimator(distributed=True)


def test_umap_affinity_distributed_true_requires_torchrun():
    """The affinity layer enforces the same launch contract as the estimator."""
    with pytest.raises(RuntimeError, match="torchrun"):
        UMAPAffinity(distributed=True)


@pytest.mark.parametrize("estimator", [UMAP, InfoTSNE])
def test_estimator_distributed_cpu_device_requires_gpu(
    estimator, initialized_cpu_group
):
    """A live group with ``device='cpu'`` is rejected with a GPU message."""
    with pytest.raises(ValueError, match="GPU"):
        estimator(distributed=True, device="cpu")


def test_umap_affinity_distributed_cpu_device_requires_gpu(initialized_cpu_group):
    """The affinity layer rejects ``device='cpu'`` under a live group."""
    with pytest.raises(ValueError, match="GPU"):
        UMAPAffinity(distributed=True, device="cpu")


@pytest.mark.parametrize("estimator", DISTRIBUTED_ESTIMATORS)
def test_estimator_default_is_not_distributed(estimator):
    """The default (``distributed='auto'``) must not raise off a launcher."""
    est = estimator()
    assert est.distributed is False


@pytest.mark.parametrize("metric", DISTRIBUTED_EVAL_METRICS)
def test_eval_metric_distributed_true_requires_torchrun(metric, no_process_group):
    with pytest.raises(RuntimeError, match="torchrun"):
        metric(distributed=True)


@pytest.mark.parametrize("metric", DISTRIBUTED_EVAL_METRICS)
def test_eval_metric_distributed_cpu_device_requires_gpu(metric, initialized_cpu_group):
    with pytest.raises(ValueError, match="GPU"):
        metric(distributed=True, device="cpu")
