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
    NegPotentialAffinity,
    NormalizedGaussianAffinity,
    NormalizedStudentAffinity,
    PACMAPAffinity,
    ScalarProductAffinity,
    SelfTuningAffinity,
    SinkhornAffinity,
    StudentAffinity,
    CauchyAffinity,
    SymmetricEntropicAffinity,
    UMAPAffinityIn,
    UMAPAffinityOut,
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
    if len(lst_backend) > 1:
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
    if len(lst_backend) > 1:
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
    if len(lst_backend) > 1:
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
    if len(lst_backend) > 1:
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
    if len(lst_backend) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_magic_affinity(dtype, metric):
    n = 10
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for backend in lst_backend:
        # Test with default alpha=1.0 (original MAGIC behavior)
        affinity = MAGICAffinity(device=DEVICE, backend=backend, metric=metric)
        P = affinity(X)
        list_P.append(P)
        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)
        check_marginal(P, one, dim=1)

    # --- check consistency between torch and keops ---
    if len(lst_backend) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)

    # Test that explicit alpha=1.0 produces the same result as default
    affinity_default = MAGICAffinity(device=DEVICE, backend=None, metric=metric)
    affinity_explicit = MAGICAffinity(
        device=DEVICE, backend=None, metric=metric, alpha=1.0
    )
    P_default = affinity_default(X)
    P_explicit = affinity_explicit(X)
    torch.testing.assert_close(P_default, P_explicit, rtol=1e-6, atol=1e-8)


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
    if len(lst_backend) > 1:
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
    if len(lst_backend) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("sparsity", [False])
@pytest.mark.parametrize("backend", lst_backend)
def test_entropic_affinity(dtype, metric, sparsity, backend):
    n = 300
    X, _ = toy_dataset(n, dtype)
    perp = 30
    tol = 1e-2  # sparse affinities do not validate the test for tol=1e-3
    zeros = torch.zeros(n, dtype=getattr(torch, dtype), device=DEVICE)
    ones = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)
    target_entropy = np.log(perp) * ones + 1

    def entropy_gap(eps, C):  # function to find the root of
        return entropy(_log_Pe(C, eps), log=True) - target_entropy

    affinity = EntropicAffinity(
        perplexity=perp,
        backend=backend,
        metric=metric,
        tol=1e-6,
        verbose=True,
        device=DEVICE,
        sparsity=sparsity,
    )
    log_P = affinity(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, backend == "keops")
    check_shape(log_P, (n, n))
    check_marginal(log_P + math.log(n), zeros, dim=1, tol=tol, log=True)
    check_entropy(log_P + math.log(n), target_entropy, dim=1, tol=tol, log=True)

    # -- check bounds on the root of entropic affinities --
    C, _ = affinity._distance_matrix(to_torch(X, device=DEVICE))
    begin, end = _bounds_entropic_affinity(C, perplexity=perp)
    assert (entropy_gap(begin, C) < 0).all(), (
        "Lower bound of entropic affinity root is not valid."
    )
    assert (entropy_gap(end, C) > 0).all(), (
        "Lower bound of entropic affinity root is not valid."
    )


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("optimizer", ["Adam", "LBFGS"])
@pytest.mark.parametrize("backend", lst_backend)
def test_sym_entropic_affinity(dtype, metric, optimizer, backend):
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
def test_doubly_stochastic_entropic(dtype, metric, backend):
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
def test_doubly_stochastic_quadratic(dtype, metric, backend):
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
    )
    P = affinity(X)

    # -- check properties of the affinity matrix --
    check_type(P, backend == "keops")
    check_shape(P, (n, n))
    check_symmetry(P)
    check_marginal(P, ones / n, dim=1, tol=tol, log=False)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("sparsity", [False])
@pytest.mark.parametrize("backend", lst_backend)
def test_umap_data_affinity(dtype, metric, sparsity, backend):
    n = 300
    X, _ = toy_dataset(n, dtype)
    n_neighbors = 30
    tol = 1e-3

    affinity = UMAPAffinityIn(
        n_neighbors=n_neighbors,
        device=DEVICE,
        backend=backend,
        metric=metric,
        tol=tol,
        verbose=True,
        sparsity=sparsity,
    )
    P = affinity(X)

    # -- check properties of the affinity matrix --
    check_type(P, backend == "keops")
    check_shape(P, (n, n))
    check_nonnegativity(P)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("backend", lst_backend)
@pytest.mark.parametrize("a, b", [(1, 2), (None, None)])
def test_umap_embedding_affinity(dtype, metric, backend, a, b):
    n = 300
    X, _ = toy_dataset(n, dtype)

    affinity = UMAPAffinityOut(
        device=DEVICE,
        backend=backend,
        metric=metric,
        verbose=True,
        a=a,
        b=b,
    )
    P = affinity(X)

    # -- check properties of the affinity matrix --
    check_type(P, backend == "keops")
    check_shape(P, (n, n))
    check_nonnegativity(P)
    check_symmetry(P)


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
    if len(lst_backend) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("backend", lst_backend)
def test_pacmap_affinity(dtype, metric, backend):
    n = 300
    X, _ = toy_dataset(n, dtype)
    n_neighbors = 30

    affinity = PACMAPAffinity(
        n_neighbors=n_neighbors,
        device=DEVICE,
        backend=backend,
        metric=metric,
        verbose=True,
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
def test_magic_alpha_decay_affinity(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    # Test with different backends using alpha=2.0 (alpha-decay mode)
    list_P_backends = []
    for backend in lst_backend:
        affinity = MAGICAffinity(
            K=7, alpha=2.0, device=DEVICE, backend=backend, metric=metric
        )
        P = affinity(X)
        list_P_backends.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_nonnegativity(P)
        check_marginal(P, one, dim=1)

    # --- check consistency between torch and keops backends ---
    if len(lst_backend) > 1:
        check_similarity_torch_keops(list_P_backends[0], list_P_backends[1], K=10)

    # Test with different alpha values to ensure they produce different results
    list_P_alpha = []
    for alpha in [1.0, 2.0]:
        affinity = MAGICAffinity(
            K=7,
            alpha=alpha,
            device=DEVICE,
            backend=None,  # Use default backend for this test
            metric=metric,
        )
        P = affinity(X)
        list_P_alpha.append(P)

    # --- check that different alpha values produce different results ---
    assert not torch.allclose(list_P_alpha[0], list_P_alpha[1], atol=1e-6), (
        "MAGIC affinities with different alpha should produce different results"
    )


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_neg_potential_affinity(dtype, metric):
    n = 30  # Use smaller n for efficiency since this is computationally intensive
    X, _ = toy_dataset(n, dtype)

    # Test with different backends
    list_P_backends = []
    for backend in lst_backend:
        affinity = NegPotentialAffinity(
            metric=metric,
            device=DEVICE,
            backend=backend,
            sigma=2.0,
            anisotropy=0.0,
            K=7,
            alpha=2.0,
            t=2,
            eps=1e-5,
        )
        P = affinity(X)
        list_P_backends.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, backend == "keops")
        check_shape(P, (n, n))
        check_symmetry(P)

        # Check that diagonal is zero (as specified in the class)
        diagonal = P.diag() if hasattr(P, "diag") else torch.diag(P)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6), (
            "Diagonal should be zero for NegPotentialAffinity"
        )

        # Check that the matrix contains negative values (negative potential distances)
        assert P.min() < 0, "NegPotentialAffinity should contain negative values"

    # --- check consistency between torch and keops backends ---
    if len(lst_backend) > 1:
        check_similarity_torch_keops(list_P_backends[0], list_P_backends[1], K=5)

    # Test with different parameter combinations to ensure they produce different results
    list_P_params = []
    for t, anisotropy in [(2, 0.0), (5, 0.5)]:
        affinity = NegPotentialAffinity(
            metric=metric,
            device=DEVICE,
            backend=None,  # Use default backend for parameter testing
            sigma=2.0,
            anisotropy=anisotropy,
            K=7,
            alpha=2.0,
            t=t,
            eps=1e-5,
        )
        P = affinity(X)
        list_P_params.append(P)

    # --- check that different parameters produce different results ---
    if len(list_P_params) > 1:
        assert not torch.allclose(list_P_params[0], list_P_params[1], atol=1e-6), (
            "Different parameters should produce different affinity matrices"
        )
