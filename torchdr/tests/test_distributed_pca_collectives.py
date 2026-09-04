"""Real-process tests for distributed PCA's collective aggregation.

These run on the Gloo CPU backend and exercise the multi-rank collectives in
:meth:`torchdr.spectral_embedding.pca.PCA._fit_transform_distributed` -- the
all-reduced global mean and covariance, and the rank-0 eigendecomposition
broadcast -- against a single-process reference.

The existing ``test_distributed_pca.py`` only pins ``world_size == 1`` with
``all_reduce``/``broadcast`` mocked to no-ops (and its mean/covariance tests
re-implement the aggregation inline rather than calling the product code), so a
one-token regression in the real aggregation is invisible to it and to any
single-process fit. Concretely, this module locks:

* the global mean ``local_sum / n_total`` -- dividing by the *local* count, or
  dropping the ``all_reduce`` of the sum, rescales the mean and is caught here;
* the global covariance ``all_reduce(local_cov)`` -- dropping it leaves every
  rank eigendecomposing only its own chunk's covariance (rank 0's is then
  broadcast to all), so the ranks still *agree* but disagree with the
  single-process components -- caught only by comparing against the reference;
* the ``broadcast(components_, src=0)`` -- caught by requiring every rank to
  return identical components.

The distributed path selects the CPU device when CUDA is unavailable, so it runs
under the ordinary two-process Gloo integration workflow with no GPU.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os

import pytest
import torch
import torch.distributed as dist

from torchdr import PCA
from torchdr.distributed import DistributedContext


pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHDR_DISTRIBUTED_TEST") != "1",
    reason="run through the dedicated multi-process integration workflow",
)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Initialize the process group created by torchrun.

    Fails loudly on a single process: these tests are meaningless unless the
    rows really cross a rank boundary, and a silent one-rank run would report
    green while all-reduce and broadcast reduce to no-ops.
    """
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    if world_size < 2:
        dist.destroy_process_group()
        pytest.fail(f"launch this module with at least two processes, got {world_size}")
    yield
    dist.barrier()
    dist.destroy_process_group()


def _dataset(n_samples, n_features, seed, dtype):
    """Deterministic anisotropic dataset, identical on every rank.

    Distinct per-feature scales keep the covariance eigenvalues well separated
    so the eigenvectors are unique up to sign, making the reference comparison
    unambiguous.
    """
    generator = torch.Generator().manual_seed(seed)
    scales = torch.arange(1, n_features + 1, dtype=torch.float64)
    data = torch.randn(n_samples, n_features, generator=generator, dtype=torch.float64)
    data = data * scales + torch.linspace(-2.0, 2.0, n_features, dtype=torch.float64)
    return data.to(dtype)


def _reference(X_full, n_components):
    """Single-process PCA a non-distributed run would produce."""
    model = PCA(n_components=n_components, distributed=False)
    embedding = model._fit_transform_standard(X_full)
    return model.mean_, model.components_, embedding


def _run_distributed_chunk(X_full, n_components):
    """Fit distributed PCA on this rank's row chunk of ``X_full``."""
    context = DistributedContext()
    n_samples = X_full.shape[0]
    chunk_start, chunk_end = context.compute_chunk_bounds(n_samples)
    assert chunk_end - chunk_start < n_samples, "rows must be split across ranks"

    model = PCA(n_components=n_components, distributed="auto")
    local_embedding = model._fit_transform_distributed(
        X_full[chunk_start:chunk_end].contiguous()
    )
    return chunk_start, chunk_end, model, local_embedding


def _align_sign_rows(candidate, reference):
    """Flip each row of ``candidate`` to match ``reference`` (PCA sign ambiguity)."""
    signs = torch.sign((candidate * reference).sum(dim=1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return candidate * signs.unsqueeze(1)


def _align_sign_columns(candidate, reference):
    """Flip each column of ``candidate`` to match ``reference``."""
    signs = torch.sign((candidate * reference).sum(dim=0))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return candidate * signs.unsqueeze(0)


def _gather_embedding(local_embedding, n_samples):
    """Concatenate every rank's chunk embedding into the full embedding."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_embedding)
    full = torch.cat(gathered, dim=0)
    assert full.shape[0] == n_samples
    return full


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "n_samples, n_features, n_components",
    [(200, 5, 3), (101, 6, 4)],
    ids=["even", "uneven"],
)
def test_distributed_pca_matches_single_process(
    n_samples, n_features, n_components, dtype
):
    """Distributed PCA reproduces the single-process mean, components, embedding.

    Each rank owns a disjoint chunk of rows; the all-reduced mean and covariance
    and the broadcast components must recover exactly what a single process would
    compute from all rows. The uneven case makes ``n_local`` differ across ranks
    so a ``/ n_local`` regression cannot coincide with ``/ n_total``.
    """
    X_full = _dataset(n_samples, n_features, seed=n_samples, dtype=dtype)
    ref_mean, ref_components, ref_embedding = _reference(X_full, n_components)

    chunk_start, chunk_end, model, local_embedding = _run_distributed_chunk(
        X_full, n_components
    )

    tol = (
        dict(rtol=1e-4, atol=1e-4)
        if dtype == torch.float32
        else dict(rtol=1e-8, atol=1e-8)
    )

    # Global mean: catches dividing by the local count or dropping the sum
    # all-reduce (both rescale the mean).
    assert model.mean_.dtype == dtype
    torch.testing.assert_close(model.mean_, ref_mean, **tol)

    # Components: catches dropping the covariance all-reduce (rank-0 local
    # eigenvectors would be broadcast instead of the global ones).
    aligned_components = _align_sign_rows(model.components_, ref_components)
    torch.testing.assert_close(aligned_components, ref_components, **tol)

    # This rank's chunk of the embedding must match the reference slice.
    full_embedding = _gather_embedding(local_embedding, n_samples)
    aligned_embedding = _align_sign_columns(full_embedding, ref_embedding)
    torch.testing.assert_close(aligned_embedding, ref_embedding, **tol)
    torch.testing.assert_close(
        aligned_embedding[chunk_start:chunk_end],
        ref_embedding[chunk_start:chunk_end],
        **tol,
    )


def test_components_identical_across_ranks():
    """The broadcast must leave every rank with the same components.

    ``eigh`` runs only on rank 0; ``broadcast(components_, src=0)`` distributes
    the result. A wrong source or a dropped broadcast would let ranks disagree,
    which a single-process test can never observe.
    """
    n_samples, n_features, n_components = 128, 5, 3
    X_full = _dataset(n_samples, n_features, seed=17, dtype=torch.float64)

    _, _, model, _ = _run_distributed_chunk(X_full, n_components)

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, model.components_)
    for other in gathered[1:]:
        torch.testing.assert_close(other, gathered[0], rtol=0.0, atol=0.0)
