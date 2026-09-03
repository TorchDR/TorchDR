"""
Tests for the distributed-usage contract guards.

UMAP, InfoTSNE and the other neighbor-embedding estimators (and their
affinities) *support* distributed execution, but only when the process group is
launched with ``torchrun``/the TorchDR CLI and the run is on GPU. Opting into
distributed mode incorrectly -- ``distributed=True`` without a live process
group, or with ``device="cpu"`` -- must fail immediately with a clear,
actionable message instead of proceeding into a broken run.

The guards live in ``AffinityBase.__init__`` and the neighbor-embedding base
``_setup_distributed``. Both are exercised here in a single CPU process: the
"requires torchrun" guard needs no process group (a normal pytest process has
``torch.distributed`` uninitialized), while the "requires GPU" guard is reached
by simulating an initialized single-rank group with mocks, mirroring
``test_distributed_pca.py``.

This is distinct from the *unsupported*-distributed rejection raised by PACMAP
and TSNEkhorn, which fail for any truthy ``distributed`` value.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

from contextlib import ExitStack
from unittest.mock import patch

import pytest

from torchdr.affinity import UMAPAffinity
from torchdr.neighbor_embedding import COSNE, SNE, TSNE, UMAP, InfoTSNE, LargeVis

# Estimators that support distributed mode. PACMAP and TSNEkhorn are excluded:
# they reject distributed unconditionally and are covered elsewhere.
DISTRIBUTED_ESTIMATORS = [UMAP, InfoTSNE, LargeVis, SNE, TSNE, COSNE]


def _simulate_initialized_group():
    """Context patching ``torch.distributed`` to look like a live 1-rank group.

    Also forces ``torch.cuda.is_available()`` to ``False`` so the guards are
    reached deterministically without touching a real device, regardless of the
    machine the test runs on.
    """
    stack = ExitStack()
    stack.enter_context(patch("torch.distributed.is_initialized", return_value=True))
    stack.enter_context(patch("torch.distributed.get_rank", return_value=0))
    stack.enter_context(patch("torch.distributed.get_world_size", return_value=1))
    stack.enter_context(patch("torch.cuda.is_available", return_value=False))
    return stack


# --- "requires torchrun": distributed=True without a live process group ---


@pytest.mark.parametrize("estimator", DISTRIBUTED_ESTIMATORS)
def test_estimator_distributed_true_requires_torchrun(estimator):
    """``distributed=True`` outside a process group raises, not proceeds."""
    with pytest.raises(RuntimeError, match="torchrun"):
        estimator(distributed=True)


def test_umap_affinity_distributed_true_requires_torchrun():
    """The affinity layer enforces the same launch contract as the estimator."""
    with pytest.raises(RuntimeError, match="torchrun"):
        UMAPAffinity(distributed=True)


# --- "requires GPU": distributed on a live group but device="cpu" ---


@pytest.mark.parametrize("estimator", [UMAP, InfoTSNE])
def test_estimator_distributed_cpu_device_requires_gpu(estimator):
    """A live group with ``device='cpu'`` is rejected with a GPU message."""
    with _simulate_initialized_group():
        with pytest.raises(ValueError, match="GPU"):
            estimator(distributed=True, device="cpu")


def test_umap_affinity_distributed_cpu_device_requires_gpu():
    """The affinity layer rejects ``device='cpu'`` under a live group."""
    with _simulate_initialized_group():
        with pytest.raises(ValueError, match="GPU"):
            UMAPAffinity(distributed=True, device="cpu")


# --- The default path stays inert without a process group ---


@pytest.mark.parametrize("estimator", DISTRIBUTED_ESTIMATORS)
def test_estimator_default_is_not_distributed(estimator):
    """The default (``distributed='auto'``) must not raise off a launcher."""
    est = estimator()
    assert est.distributed is False
