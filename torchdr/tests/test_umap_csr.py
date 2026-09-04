"""Tests for UMAP's flat (CSR-style) attractive-gradient representation.

UMAP flattens the max-degree-padded ``(chunk, max_degree)`` neighbor grid into
flat per-edge ``source``/``target`` buffers and computes the closed-form
attractive gradient with a deterministic segment reduction instead of a
padded ``einsum``. These tests pin that flattening and verify the segment
reduction against an independent brute-force reference, on CPU (run in CI) and,
when available, on CUDA (opt-in device coverage; the campaign's target setting).
"""

# License: BSD 3-Clause License

import numpy as np
import pytest
import torch

from torchdr.neighbor_embedding import UMAP
from torchdr.tests.utils import toy_dataset


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_flatten_padded_edges_row_major():
    """Padded grid flattens to row-major real edges with a sorted source."""
    # max_degree 4; rows with 2, 4, 0 and 1 real edges (-1 is padding).
    nn = torch.tensor(
        [
            [3, 5, -1, -1],
            [0, 1, 2, 4],
            [-1, -1, -1, -1],
            [7, -1, -1, -1],
        ]
    )
    source, target, mask = UMAP._flatten_padded_edges(nn)

    assert source.tolist() == [0, 0, 1, 1, 1, 1, 3]
    assert target.tolist() == [3, 5, 0, 1, 2, 4, 7]
    assert mask.tolist() == (nn >= 0).tolist()
    # source is monotonic non-decreasing (contiguous per-row segments): this is
    # what lets an aligned per-edge tensor be gathered the same way ...
    assert torch.all(source[1:] >= source[:-1])
    # ... e.g. the affinity values flatten by the very same mask.
    aff = torch.arange(nn.numel(), dtype=torch.float32).reshape(nn.shape)
    assert aff[mask].tolist() == [0, 1, 4, 5, 6, 7, 12]


def test_flatten_padded_edges_all_padded_row_is_dropped():
    """A fully-padded row contributes no edges (an isolated node)."""
    nn = torch.tensor([[-1, -1], [2, 3]])
    source, target, mask = UMAP._flatten_padded_edges(nn)
    assert source.tolist() == [1, 1]
    assert target.tolist() == [2, 3]
    assert mask.sum().item() == 2


def _reference_attractive_grad(emb, chunk_indices, source, target, a, b, sampled):
    """Brute-force per-edge accumulation of the UMAP attractive gradient.

    Mirrors the closed-form coefficient in ``_compute_attractive_gradients`` but
    accumulates edge by edge in Python, independent of ``segment_reduce``.
    """
    n_rows = chunk_indices.shape[0]
    grad = torch.zeros((n_rows, emb.shape[1]), dtype=emb.dtype, device=emb.device)
    for e in range(source.shape[0]):
        s_local = int(source[e])
        d = emb[int(chunk_indices[s_local])] - emb[int(target[e])]
        dist2 = float((d * d).sum())
        if dist2 > 0 and bool(sampled[e]):
            coef = (2 * a * b * dist2 ** (b - 1)) / (1 + a * dist2**b)
        else:
            coef = 0.0
        grad[s_local] += d * coef
    return grad.clamp_(-4, 4)


@pytest.mark.parametrize("device", DEVICES)
def test_attractive_gradient_matches_bruteforce_reference(device):
    """The flat segment-reduced gradient equals an edge-by-edge reference.

    A subset of edges is left unsampled (schedule) and one edge is degenerate
    (zero distance) so both masking branches are exercised.
    """
    n, dim, a, b = 9, 2, 1.577, 0.895
    gen = torch.Generator().manual_seed(0)
    emb = torch.randn(n, dim, dtype=torch.float64, generator=gen)
    chunk_indices = torch.arange(2, 8)
    emb[6] = emb[2]  # a zero-distance edge (6 -> 2) must contribute nothing

    # The six local rows represent global rows 2..7. A non-zero chunk offset
    # ensures the test catches confusion between local edge sources and global
    # embedding rows, which only affects distributed fits.
    nn = torch.tensor(
        [
            [3, 4, -1],
            [2, 5, 6],
            [-1, -1, -1],
            [7, -1, -1],
            [2, -1, -1],
            [3, 4, 5],
        ]
    )
    source, target, _ = UMAP._flatten_padded_edges(nn)

    # Sample every other edge to exercise the epoch-schedule masking.
    n_edges = source.shape[0]
    sampled = torch.zeros(n_edges, dtype=torch.bool)
    sampled[::2] = True

    model = UMAP(n_components=dim, n_neighbors=2, optimizer="SGD")
    model._a, model._b, model.n_iter_ = a, b, 0
    model.embedding_ = emb.to(device)
    model.chunk_indices_ = chunk_indices.to(device)
    model.attractive_source_ = source.to(device)
    model.attractive_target_ = target.to(device)
    model.attractive_counts_ = torch.bincount(source, minlength=len(chunk_indices)).to(
        device
    )
    # epoch_of_next_sample <= n_iter_ + 1 selects an edge this step; push the
    # unsampled edges just past the threshold.
    epoch_next = torch.where(
        sampled, torch.ones(n_edges), torch.full((n_edges,), 5.0)
    ).to(device)
    model.epoch_of_next_sample = epoch_next.clone()
    model.epochs_per_sample = torch.ones(n_edges, device=device)

    grad = model._compute_attractive_gradients()

    grad_ref = _reference_attractive_grad(
        emb.to(device),
        chunk_indices.to(device),
        source.to(device),
        target.to(device),
        a,
        b,
        sampled.to(device),
    )
    torch.testing.assert_close(grad, grad_ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_repulsive_gradient_matches_bruteforce_reference(device):
    """Repulsive gradients use global query rows for a non-zero chunk."""
    a, b, eps = 1.577, 0.895, 1e-3
    gen = torch.Generator().manual_seed(1)
    emb = torch.randn(8, 2, dtype=torch.float64, generator=gen).to(device)
    chunk_indices = torch.tensor([3, 4, 5], device=device)
    neg_indices = torch.tensor([[0, 7, 2], [6, 1, 3], [1, 7, 0]], device=device)

    model = UMAP(n_components=2, n_neighbors=2, optimizer="SGD")
    model._a, model._b, model._eps = a, b, eps
    model.embedding_ = emb
    model.chunk_indices_ = chunk_indices
    model.neg_indices_ = neg_indices
    model.n_negatives = neg_indices.shape[1]
    model.negative_sample_rate = 1
    model.attractive_counts_ = torch.tensor([2, 1, 3], device=device)
    model.mask_affinity_in_ = torch.tensor(
        [True, False, True, True, False, True], device=device
    )

    grad = model._compute_repulsive_gradients()

    # The active positive-edge counts retain 1, 1, and 2 negative samples.
    grad_ref = torch.zeros_like(grad)
    for local_row, n_negatives in enumerate([1, 1, 2]):
        for negative in neg_indices[local_row, :n_negatives]:
            diff = emb[chunk_indices[local_row]] - emb[negative]
            dist2 = (diff * diff).sum()
            coefficient = -2 * b / ((dist2 + eps) * (1 + a * dist2**b))
            grad_ref[local_row] += coefficient * diff
    grad_ref.clamp_(-4, 4)

    torch.testing.assert_close(grad, grad_ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_umap_fit_is_reproducible(device):
    """The CSR path remains reproducible with a fixed seed."""
    X, _ = toy_dataset(80, "float32")
    kw = dict(
        n_components=2,
        backend=None,
        device=device,
        init="normal",
        max_iter=40,
        random_state=0,
        n_neighbors=8,
        optimizer="SGD",
    )
    a = torch.as_tensor(UMAP(**kw).fit_transform(X))
    b = torch.as_tensor(UMAP(**kw).fit_transform(X))
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_umap_fit_cuda_smoke():
    """UMAP fits on CUDA and returns a finite embedding (device path exercised)."""
    X, _ = toy_dataset(200, "float32")
    model = UMAP(
        n_components=2,
        device="cuda",
        init="normal",
        max_iter=50,
        random_state=0,
        n_neighbors=10,
        optimizer="SGD",
    )
    # Be robust to a global deterministic mode that another test may have left
    # enabled: UMAP's repulsive einsum uses cuBLAS, which raises under
    # deterministic algorithms unless CUBLAS_WORKSPACE_CONFIG is set.
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        emb = torch.as_tensor(model.fit_transform(X))
    finally:
        torch.use_deterministic_algorithms(prev)
    assert emb.shape == (200, 2)
    assert np.isfinite(emb.cpu().numpy()).all()
