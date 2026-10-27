"""Independent references and workflows for multi-perplexity affinities."""

import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from torchdr import MultiscaleEntropicAffinity
from torchdr.affinity import EntropicAffinity


def data(dtype=torch.float64):
    return torch.randn(40, 4, generator=torch.Generator().manual_seed(31), dtype=dtype)


def reference(X, p):
    D = cdist(X, X, metric="sqeuclidean")
    np.fill_diagonal(D, np.inf)
    P = np.zeros_like(D)
    for i in range(len(X)):
        ids = np.flatnonzero(np.isfinite(D[i]))

        def lp(beta, row=i, valid_ids=ids):
            q = -np.exp(beta) * D[row, valid_ids]
            return q - logsumexp(q)

        beta = brentq(lambda b: -np.dot(np.exp(lp(b)), lp(b)) - np.log(p), -25, 25)
        P[i, ids] = np.exp(lp(beta)) / len(X)
    return P


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_independent_reference(dtype):
    X = data(dtype)
    model = MultiscaleEntropicAffinity(
        (4, 9), sparsity=False, device="cpu", distributed=False
    )
    P = model(X, return_indices=False)
    expected = (reference(X.numpy(), 4) + reference(X.numpy(), 9)) / 2
    np.testing.assert_allclose(P.numpy(), expected, atol=2e-7)
    torch.testing.assert_close(P.sum(1), torch.full((40,), 1 / 40, dtype=dtype))
    assert torch.equal(P.diag(), torch.zeros(40, dtype=dtype))
    torch.testing.assert_close(model(X, log=True, return_indices=False).exp(), P)
    assert model.eps_.shape == (2, 40)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("scales", [(4, 9), (4, 9, 4)])
def test_single_scale_and_sparse_support(sparse, scales):
    X = data()
    combined, ids = MultiscaleEntropicAffinity(
        scales, sparsity=sparse, device="cpu", distributed=False
    )(X)
    expected = torch.zeros((40, 40), dtype=X.dtype)
    for p in scales:
        P, idx = EntropicAffinity(p, sparsity=sparse, device="cpu", distributed=False)(
            X
        )
        single, single_ids = MultiscaleEntropicAffinity(
            (p,), sparsity=sparse, device="cpu", distributed=False
        )(X)
        torch.testing.assert_close(single, P)
        if sparse:
            torch.testing.assert_close(single_ids, idx)
            expected.scatter_add_(1, idx.to(dtype=torch.long), P / len(scales))
        else:
            expected += P / len(scales)
    actual = (
        torch.zeros_like(expected).scatter_(1, ids.to(dtype=torch.long), combined)
        if sparse
        else combined
    )
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-4)


def test_order_duplicate_scales_and_shared_search(monkeypatch):
    X = data()
    model = MultiscaleEntropicAffinity((4, 9), device="cpu", distributed=False)
    calls = []
    original = model._distance_matrix

    def distance(*args, **kwargs):
        calls.append(kwargs.get("k"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_distance_matrix", distance)
    P, ids = model(X)
    assert len(calls) == 1
    assert int(calls[0]) == 27
    for scales in ((9, 4), (4, 9, 4, 9)):
        other, idx = MultiscaleEntropicAffinity(
            scales, device="cpu", distributed=False
        )(X)
        torch.testing.assert_close(other, P)
        torch.testing.assert_close(idx, ids)


@pytest.mark.parametrize(
    "scales", [(), 2, (1, 4), (True, 4), (np.nan, 4), (np.inf, 4), (3.5, 4), ("4", 9)]
)
def test_invalid_scales(scales):
    with pytest.raises(ValueError):
        MultiscaleEntropicAffinity(scales)


def test_dense_embedding_workflow():
    from torchdr import AffinityMatcher
    from torchdr.affinity import NormalizedStudentAffinity

    X = data()
    model = AffinityMatcher(
        affinity_in=MultiscaleEntropicAffinity(
            (4, 9), sparsity=False, device="cpu", distributed=False
        ),
        affinity_out=NormalizedStudentAffinity(device="cpu"),
        loss_fn="cross_entropy_loss",
        n_components=2,
        max_iter=20,
        compile=False,
        device="cpu",
        random_state=13,
        lr=0.1,
        check_interval=5,
    )
    embedding = model.fit_transform(X)
    assert embedding.shape == (40, 2)
    assert torch.isfinite(embedding).all()
    P = model.affinity_in(X, return_indices=False)
    torch.testing.assert_close(P.sum(), torch.tensor(1.0, dtype=X.dtype))


@pytest.mark.parametrize("sparsity", [False, True])
def test_compile_matches_eager_with_eager_backend(monkeypatch, caplog, sparsity):
    original_compile = torch.compile
    compile_calls = []

    def compile_with_eager_backend(function):
        compile_calls.append(function)
        return original_compile(function, backend="eager")

    monkeypatch.setattr(torch, "compile", compile_with_eager_backend)
    X = data()
    eager = MultiscaleEntropicAffinity(
        (4, 9), device="cpu", distributed=False, compile=False, sparsity=sparsity
    )
    compiled = MultiscaleEntropicAffinity(
        (4, 9), device="cpu", distributed=False, compile=True, sparsity=sparsity
    )

    expected, expected_indices = eager(X)
    actual, actual_indices = compiled(X)

    torch.testing.assert_close(actual, expected)
    if expected_indices is None:
        assert actual_indices is None
    else:
        torch.testing.assert_close(actual_indices, expected_indices)
    assert len(compile_calls) == 1
    assert "Could not compile" not in caplog.text


def test_affinity_matcher_compile_enables_multiscale_affinity(monkeypatch, caplog):
    from torchdr import AffinityMatcher
    from torchdr.affinity import NormalizedStudentAffinity

    original_compile = torch.compile
    compile_calls = []

    def compile_with_eager_backend(function):
        compile_calls.append(function)
        return original_compile(function, backend="eager")

    monkeypatch.setattr(torch, "compile", compile_with_eager_backend)
    X = data()
    affinity_in = MultiscaleEntropicAffinity(
        (4, 9), sparsity=False, device="cpu", distributed=False
    )
    model = AffinityMatcher(
        affinity_in=affinity_in,
        affinity_out=NormalizedStudentAffinity(device="cpu"),
        loss_fn="cross_entropy_loss",
        n_components=2,
        max_iter=0,
        compile=True,
        device="cpu",
        random_state=13,
    )

    assert affinity_in.compile
    embedding = model.fit_transform(X)

    assert embedding.shape == (40, 2)
    assert torch.isfinite(embedding).all()
    assert len(compile_calls) == 1
    assert "Could not compile" not in caplog.text


def test_log_domain_keeps_small_probabilities_finite():
    X = data()
    log_P = MultiscaleEntropicAffinity(
        (2, 3), sparsity=False, device="cpu", distributed=False
    )(X, log=True, return_indices=False)
    mask = ~torch.eye(len(X), dtype=torch.bool)
    assert torch.isfinite(log_P[mask]).all()
    assert torch.equal(log_P.diag().exp(), torch.zeros(len(X), dtype=X.dtype))


def test_sparse_faiss_matches_exact_tensor_workflow():
    pytest.importorskip("faiss")
    X = data(torch.float32)
    tensor, ti = MultiscaleEntropicAffinity((4, 9), device="cpu", distributed=False)(X)
    indexed, fi = MultiscaleEntropicAffinity(
        (4, 9), backend="faiss", device="cpu", distributed=False
    )(X)
    full_tensor = torch.zeros((40, 40)).scatter_(1, ti, tensor)
    full_indexed = torch.zeros((40, 40)).scatter_(1, fi, indexed)
    torch.testing.assert_close(full_indexed, full_tensor, atol=2e-7, rtol=1e-4)
    torch.testing.assert_close(full_indexed.sum(1), torch.full((40,), 1 / 40))


@pytest.mark.parametrize(
    "keyword",
    [
        {"backend": "keops"},
        {"backend": "faiss", "compile": True},
    ],
)
def test_unsupported_paths_fail_explicitly(keyword):
    with pytest.raises(ValueError):
        MultiscaleEntropicAffinity((4, 9), **keyword)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_too_few_samples(n):
    with pytest.raises(ValueError, match="at least 4 samples"):
        MultiscaleEntropicAffinity((2,))(torch.zeros(n, 2))


def test_input_size_clamping_and_numpy_conversion():
    X = data().numpy()[:12]
    P = MultiscaleEntropicAffinity(
        (4, 30), sparsity=False, device="cpu", distributed=False
    )(X, return_indices=False)
    expected = MultiscaleEntropicAffinity(
        (4, 10), sparsity=False, device="cpu", distributed=False
    )(X, return_indices=False)
    torch.testing.assert_close(P, expected)
    torch.testing.assert_close(P.sum(1), torch.full((12,), 1 / 12, dtype=P.dtype))
