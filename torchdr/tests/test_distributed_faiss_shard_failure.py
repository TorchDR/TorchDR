"""Real-process test that a sharded-search failure propagates without a hang.

Issue #301 requires that collective failures in the sharded search reach every
rank instead of deadlocking. The sharded loop issues an identical
broadcast/all_gather sequence on each rank, so when one rank drops out of that
sequence -- here by a collective raising on rank 0 -- the peers waiting on the
matching collective observe the closed connection and raise as well, rather than
blocking forever.

This module owns its process group and creates it with a short timeout: the
sibling sharded-search module relies on ``init_distributed``'s default timeout,
which is far longer than a CI job, whereas a propagation test must give up
quickly if a platform ever failed to surface the dropped peer. In practice the
peers raise in milliseconds from the closed connection, well before the timeout.

Ordinary CI runs this against Gloo and CPU FAISS, matching the sibling sharded
search module. The NCCL backend provides the same guarantee through its own
collective watchdog rather than an immediate connection reset.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os
import time
from datetime import timedelta
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import torchdr.distance.faiss as faiss_mod
from torchdr.distance.faiss import sharded_pairwise_distances_faiss
from torchdr.distributed import DistributedContext
from torchdr.utils import faiss


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
        reason="run through the dedicated multi-process integration workflow",
    ),
    pytest.mark.skipif(faiss is None or faiss is False, reason="faiss not installed"),
]

N_SAMPLES = 400
N_FEATURES = 8
K = 5
# Safety net only: the peers normally raise in milliseconds from the closed
# connection. If that were ever missed, the group still gives up this far below
# the CI job budget instead of hanging.
PG_TIMEOUT_S = 60
# A hang would blow past this; a propagated failure stays orders of magnitude
# under it.
PROPAGATION_CEILING_S = 45


@pytest.fixture(scope="module", autouse=True)
def short_timeout_process_group():
    """Create a Gloo process group with an explicit short timeout.

    Fails loudly on a single process: with one rank there is no peer to keep
    waiting, so the deadlock this module guards against cannot occur and a
    one-rank run would report green while exercising nothing.
    """
    if dist.is_initialized():
        pytest.fail("expected an uninitialized process group for this module")
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=PG_TIMEOUT_S))
    world_size = dist.get_world_size()
    if world_size < 2:
        dist.destroy_process_group()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    # The group is intentionally left broken by the test; tear down best-effort.
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture(scope="module")
def context(short_timeout_process_group):
    return DistributedContext()


@pytest.fixture(scope="module")
def data():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N_SAMPLES, N_FEATURES, generator=generator)


def test_collective_failure_propagates_without_deadlock(data, context):
    # Rank 0 takes part in the first broadcast, then a collective raises on it,
    # standing in for any per-rank failure at a collective (an OOM, a shape
    # mismatch, a dead GPU). The peers are blocked in the matching collective and
    # must raise rather than wait forever.
    if context.rank == 0:
        with mock.patch.object(
            faiss_mod.dist,
            "all_gather",
            side_effect=RuntimeError("injected collective failure on rank 0"),
        ):
            with pytest.raises(RuntimeError, match="injected collective failure"):
                sharded_pairwise_distances_faiss(
                    data, k=K, metric="sqeuclidean", distributed_ctx=context
                )
        # Closing this rank's connections lets the peers observe the drop at once
        # instead of waiting out the group timeout.
        dist.destroy_process_group()
    else:
        start = time.perf_counter()
        with pytest.raises(RuntimeError):
            sharded_pairwise_distances_faiss(
                data, k=K, metric="sqeuclidean", distributed_ctx=context
            )
        elapsed = time.perf_counter() - start
        assert elapsed < PROPAGATION_CEILING_S, (
            f"peer rank {context.rank} took {elapsed:.1f}s to observe the "
            f"failure; the collective is deadlocking rather than propagating"
        )
        # Drop this rank's now-broken group so module teardown has nothing to do.
        dist.destroy_process_group()
