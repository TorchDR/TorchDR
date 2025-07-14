"""
Tests for affinity matrices.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

import math

import numpy as np
import pytest
import torch

from torchdr.utils import pykeops

# define lists for keops testing
if pykeops:
    lst_backend = [None, "keops"]
else:
    pykeops = False
    lst_backend = [None]


from torchdr.affinity import (
    DoublyStochasticQuadraticAffinity,
    EntropicAffinity,
    GaussianAffinity,
    MAGICAffinity,
    NegativeCostAffinity,
    NormalizedGaussianAffinity,
    NormalizedStudentAffinity,
    PACMAPAffinity,
    PHATEAffinity,
    ScalarProductAffinity,
    SelfTuningAffinity,
    SinkhornAffinity,
    StudentAffinity,
    CauchyAffinity,
    SymmetricEntropicAffinity,
    UMAPAffinity,
)
from torchdr.affinity.entropic import _bounds_entropic_affinity, _log_Pe
from torchdr.tests.utils import toy_dataset
from torchdr.utils import (
    check_entropy,
    check_marginal,
    check_nonnegativity,
    check_shape,
    check_similarity_torch_keops,
    check_symmetry,
    check_total_sum,
    check_type,
    entropy,
    to_torch,
)

lst_types = ["float32", "float64"]

LIST_METRICS_TEST = ["sqeuclidean"]
DEVICE = "cpu"


@pytest.mark.parametrize("dtype", lst_types)
def test_scalar_product_affinity(dtype):
    n = 50
    X, _ = toy_dataset(n, dtype)

    list_P = []
    for backend in lst_backend:
        affinity = ScalarProductAffinity(device=DEVICE, backend=backend)
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_symmetry(P)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_normalized_gibbs_affinity(dtype, metric, dim):
    n = 50
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for backend in lst_backend:
        affinity = NormalizedGaussianAffinity(
            device=DEVICE, backend=backend, metric=metric, normalization_dim=dim
        )
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)
        if isinstance(dim, int):
            check_marginal(P * n, one, dim=dim)
        else:
            check_total_sum(P, 1)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_normalized_student_affinity(dtype, metric, dim):
    n = 50
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for backend in lst_backend:
        affinity = NormalizedStudentAffinity(
            device=DEVICE, backend=backend, metric=metric, normalization_dim=dim
        )
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)
        if isinstance(dim, int):
            check_marginal(P * n, one, dim=dim)
        else:
            check_total_sum(P, 1)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_gibbs_affinity(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)

    list_P = []
    for backend in lst_backend:
        affinity = GaussianAffinity(device=DEVICE, backend=backend, metric=metric)
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_self_tuning_gibbs_affinity(dtype, metric, dim):
    n = 10
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for backend in lst_backend:
        affinity = SelfTuningAffinity(
            device=DEVICE, backend=backend, metric=metric, normalization_dim=dim
        )
        P = affinity(X)
        list_P.append(P)
        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)
        if isinstance(dim, int):
            check_marginal(P, one, dim=dim)
        else:
            check_total_sum(P, 1)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_magic_affinity(dtype, metric):
    n = 10
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for backend in lst_backend:
        # Test MAGIC affinity
        affinity = MAGICAffinity(device=DEVICE, backend=backend, metric=metric)
        P = affinity(X)
        list_P.append(P)
        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)
        check_marginal(P, one, dim=1)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_student_affinity(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)

    list_P = []
    for backend in lst_backend:
        affinity = StudentAffinity(device=DEVICE, backend=backend, metric=metric)
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_cauchy_affinity(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)

    list_P = []
    for backend in lst_backend:
        affinity = CauchyAffinity(device=DEVICE, backend=backend, metric=metric)
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("sparsity", [False, True])
@pytest.mark.parametrize("backend", lst_backend)
def test_entropic_affinity(dtype, metric, sparsity, backend, compile=False):
    if backend is not None and compile:
        pytest.skip("torch.compile is only supported for backend=None")

    n = 300
    X, _ = toy_dataset(n, dtype)
    perp = 30
    # tol = 1e-2  # sparse affinities do not validate the test for tol=1e-3
    zeros = torch.zeros(n, dtype=getattr(torch, dtype), device=DEVICE)
    ones = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)
    target_entropy = np.log(perp) * ones + 1

    def entropy_gap(eps, C):  # function to find the root of
        return entropy(_log_Pe(C, eps), log=True) - target_entropy

    # test with a given perplexity
    affinity = EntropicAffinity(
        perplexity=perp,
        backend=backend,
        metric=metric,
        verbose=True,
        device=DEVICE,
        sparsity=sparsity,
        compile=compile,
    )
    log_P = affinity(X, log=True, return_indices=False)

    tol = 1e-3
    # -- check properties of the affinity matrix --
    if not sparsity:
        check_type(log_P, backend == "keops")
        check_shape(log_P, (n, n))
    check_marginal(log_P + math.log(n), zeros, dim=1, tol=tol, log=True)
    check_entropy(log_P + math.log(n), target_entropy, dim=1, tol=tol, log=True)

    # -- check bounds on the root of entropic affinities --
    C, _ = affinity._distance_matrix(to_torch(X, device=DEVICE))
    perp_tensor = torch.tensor(perp, dtype=getattr(torch, dtype), device=DEVICE)
    begin, end = _bounds_entropic_affinity(
        C, perplexity=perp_tensor, device=DEVICE, dtype=getattr(torch, dtype)
    )
    assert (entropy_gap(begin, C) < 0).all(), (
        "Lower bound of entropic affinity root is not valid."
    )
    assert (entropy_gap(end, C) > 0).all(), (
        "Upper bound of entropic affinity root is not valid."
    )


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("optimizer", ["Adam", "LBFGS"])
@pytest.mark.parametrize("backend", lst_backend)
def test_sym_entropic_affinity(dtype, metric, optimizer, backend, compile=False):
    if backend is not None and compile:
        pytest.skip("torch.compile is only supported for backend=None")

    n = 300
    X, _ = toy_dataset(n, dtype)
    perp = 30
    tol = 1e-2
    zeros = torch.zeros(n, dtype=getattr(torch, dtype), device=DEVICE)
    ones = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)
    lr = 1e0 if optimizer == "LBFGS" else 1e-1

    affinity = SymmetricEntropicAffinity(
        perplexity=perp,
        backend=backend,
        metric=metric,
        tol=1e-5,
        verbose=True,
        lr=lr,
        max_iter=3000,
        eps_square=True,
        device=DEVICE,
        optimizer=optimizer,
        compile=compile,
    )
    log_P = affinity(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, backend == "keops")
    check_shape(log_P, (n, n))
    check_symmetry(log_P)
    check_marginal(log_P + math.log(n), zeros, dim=1, tol=tol, log=True)
    check_entropy(
        log_P + math.log(n), np.log(perp) * ones + 1, dim=1, tol=tol, log=True
    )


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("backend", lst_backend)
def test_doubly_stochastic_entropic(dtype, metric, backend, compile=False):
    if backend is not None and compile:
        pytest.skip("torch.compile is only supported for backend=None")

    n = 300
    X, _ = toy_dataset(n, dtype)
    eps = 1e0
    tol = 1e-3
    zeros = torch.zeros(n, dtype=getattr(torch, dtype), device=DEVICE)

    affinity = SinkhornAffinity(
        eps=eps,
        backend=backend,
        metric=metric,
        tol=tol,
        device=DEVICE,
        verbose=True,
        compile=compile,
    )
    log_P = affinity(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, backend == "keops")
    check_shape(log_P, (n, n))
    check_symmetry(log_P)
    check_marginal(log_P + math.log(n), zeros, dim=1, tol=tol, log=True)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("backend", lst_backend)
def test_doubly_stochastic_quadratic(dtype, metric, backend, compile=False):
    if backend is not None and compile:
        pytest.skip("torch.compile is only supported for backend=None")

    n = 300
    X, _ = toy_dataset(n, dtype)
    eps = 1e0
    tol = 1e-3
    ones = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    affinity = DoublyStochasticQuadraticAffinity(
        eps=eps,
        backend=backend,
        metric=metric,
        tol=tol,
        device=DEVICE,
        verbose=True,
        compile=compile,
    )
    P = affinity(X)

    # -- check properties of the affinity matrix --
    check_type(P, backend == "keops")
    check_shape(P, (n, n))
    check_symmetry(P)
    check_marginal(P, ones / n, dim=1, tol=tol, log=False)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("sparsity", [True, False])
@pytest.mark.parametrize("backend", lst_backend)
def test_umap_data_affinity(dtype, metric, sparsity, backend, compile=False):
    if backend is not None and compile:
        pytest.skip("torch.compile is only supported for backend=None")

    n = 300
    X, _ = toy_dataset(n, dtype)
    n_neighbors = 30

    affinity = UMAPAffinity(
        n_neighbors=n_neighbors,
        device=DEVICE,
        backend=backend,
        metric=metric,
        verbose=True,
        sparsity=sparsity,
        compile=compile,
    )
    P = affinity(X, return_indices=False)

    # -- check properties of the affinity matrix --
    if not sparsity:
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
    check_nonnegativity(P)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_negative_cost_affinity(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)

    list_P = []
    for backend in lst_backend:
        affinity = NegativeCostAffinity(device=DEVICE, backend=backend, metric=metric)
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_symmetry(P)

    # --- check consistency between torch and keops ---
    if "keops" in lst_backend:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("backend", lst_backend)
def test_pacmap_affinity(dtype, metric, backend, compile=False):
    if backend is not None and compile:
        pytest.skip("torch.compile is only supported for backend=None")

    n = 300
    X, _ = toy_dataset(n, dtype)
    n_neighbors = 30

    affinity = PACMAPAffinity(
        n_neighbors=n_neighbors,
        device=DEVICE,
        backend=backend,
        metric=metric,
        verbose=True,
        compile=compile,
    )

    # PACMAPAffinity returns None for affinities and only indices
    _, indices = affinity(X, return_indices=True)

    # -- check properties of the neighborhood indices --
    assert isinstance(indices, torch.Tensor), "Indices should be a torch.Tensor"
    assert indices.shape == (n, n_neighbors), (
        f"Expected shape {(n, n_neighbors)}, got {indices.shape}"
    )
    assert indices.dtype == torch.int, f"Expected dtype torch.int, got {indices.dtype}"

    # Check that no index exceeds the number of samples
    assert torch.all(indices >= 0) and torch.all(indices < n), "Indices out of bounds"

    # Check that rho_ was correctly computed
    assert hasattr(affinity, "rho_"), "rho_ attribute not found"
    assert affinity.rho_.shape == (n,), (
        f"Expected rho_ shape {(n,)}, got {affinity.rho_.shape}"
    )
    assert torch.all(affinity.rho_ > 0), "rho_ values should be positive"


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_phate_affinity(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)

    # PHATE doesn't support keops/faiss backends, so we only test with None backend
    affinity = PHATEAffinity(
        device=DEVICE, backend=None, metric=metric, k=5, alpha=2.0, t=3, compile=False
    )
    P = affinity(X)

    # -- check properties of the affinity matrix --
    check_type(P, False)  # backend is always None, so LazyTensor is False
    check_shape(P, (n, n))
    check_symmetry(P)

    # Check that sigma_ was computed
    assert hasattr(affinity, "sigma_"), "sigma_ attribute not found"
    assert affinity.sigma_.shape == (n,), (
        f"Expected sigma_ shape {(n,)}, got {affinity.sigma_.shape}"
    )
    assert torch.all(affinity.sigma_ > 0), "sigma_ values should be positive"
