# -*- coding: utf-8 -*-
"""
Tests for affinity matrices.
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch
import numpy as np
import math

from torchdr.utils import pykeops

# define lists for keops testing
if pykeops:
    lst_keops = [False, True]
else:
    pykeops = False
    lst_keops = [False]


from torchdr.utils import (
    check_similarity_torch_keops,
    check_symmetry,
    check_marginal,
    check_entropy,
    check_type,
    check_shape,
    check_nonnegativity,
    check_total_sum,
    entropy,
    to_torch,
)
from torchdr.affinity import (
    ScalarProductAffinity,
    GaussianAffinity,
    NormalizedGaussianAffinity,
    NormalizedStudentAffinity,
    SelfTuningAffinity,
    MAGICAffinity,
    StudentAffinity,
    EntropicAffinity,
    SymmetricEntropicAffinity,
    SinkhornAffinity,
    DoublyStochasticQuadraticAffinity,
    UMAPAffinityIn,
    UMAPAffinityOut,
)
from torchdr.affinity.entropic import _bounds_entropic_affinity, _log_Pe
from torchdr.tests.utils import toy_dataset

lst_types = ["float32", "float64"]

LIST_METRICS_TEST = ["sqeuclidean"]
DEVICE = "cpu"


@pytest.mark.skipif(pykeops, reason="pykeops is available")
def test_keops_not_installed():
    with pytest.raises(ValueError, match="pykeops is not installed"):
        ScalarProductAffinity(keops=True)


@pytest.mark.parametrize("dtype", lst_types)
def test_scalar_product_affinity(dtype):
    n = 50
    X, _ = toy_dataset(n, dtype)

    list_P = []
    for keops in lst_keops:
        affinity = ScalarProductAffinity(device=DEVICE, keops=keops)
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_symmetry(P)

    # --- check consistency between torch and keops ---
    if len(lst_keops) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_normalized_gibbs_affinity(dtype, metric, dim):
    n = 50
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for keops in lst_keops:
        affinity = NormalizedGaussianAffinity(
            device=DEVICE, keops=keops, metric=metric, normalization_dim=dim
        )
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_nonnegativity(P)
        if isinstance(dim, int):
            check_marginal(P * n, one, dim=dim)
        else:
            check_total_sum(P, 1)

    # --- check consistency between torch and keops ---
    if len(lst_keops) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_normalized_student_affinity(dtype, metric, dim):
    n = 50
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for keops in lst_keops:
        affinity = NormalizedStudentAffinity(
            device=DEVICE, keops=keops, metric=metric, normalization_dim=dim
        )
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_nonnegativity(P)
        if isinstance(dim, int):
            check_marginal(P * n, one, dim=dim)
        else:
            check_total_sum(P, 1)

    # --- check consistency between torch and keops ---
    if len(lst_keops) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_gibbs_affinity(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)

    list_P = []
    for keops in lst_keops:
        affinity = GaussianAffinity(device=DEVICE, keops=keops, metric=metric)
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_nonnegativity(P)

    # --- check consistency between torch and keops ---
    if len(lst_keops) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_self_tuning_gibbs_affinity(dtype, metric, dim):
    n = 10
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for keops in lst_keops:
        affinity = SelfTuningAffinity(
            device=DEVICE, keops=keops, metric=metric, normalization_dim=dim
        )
        P = affinity(X)
        list_P.append(P)
        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_nonnegativity(P)
        if isinstance(dim, int):
            check_marginal(P, one, dim=dim)
        else:
            check_total_sum(P, 1)

    # --- check consistency between torch and keops ---
    if len(lst_keops) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_magic_affinity(dtype, metric):
    n = 10
    X, _ = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for keops in lst_keops:
        affinity = MAGICAffinity(device=DEVICE, keops=keops, metric=metric)
        P = affinity(X)
        list_P.append(P)
        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_nonnegativity(P)
        check_marginal(P, one, dim=1)

    # --- check consistency between torch and keops ---
    if len(lst_keops) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
def test_student_affinity(dtype, metric):
    n = 50
    X, _ = toy_dataset(n, dtype)

    list_P = []
    for keops in lst_keops:
        affinity = StudentAffinity(device=DEVICE, keops=keops, metric=metric)
        P = affinity(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_nonnegativity(P)

    # --- check consistency between torch and keops ---
    if len(lst_keops) > 1:
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("sparsity", [False])
@pytest.mark.parametrize("keops", lst_keops)
def test_entropic_affinity(dtype, metric, sparsity, keops):
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
        keops=keops,
        metric=metric,
        tol=1e-6,
        verbose=True,
        device=DEVICE,
        sparsity=sparsity,
    )
    log_P = affinity(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, keops=keops)
    check_shape(log_P, (n, n))
    check_marginal(log_P + math.log(n), zeros, dim=1, tol=tol, log=True)
    check_entropy(log_P + math.log(n), target_entropy, dim=1, tol=tol, log=True)

    # -- check bounds on the root of entropic affinities --
    C = affinity._distance_matrix(to_torch(X, device=DEVICE))
    begin, end = _bounds_entropic_affinity(C, perplexity=perp)
    assert (
        entropy_gap(begin, C) < 0
    ).all(), "Lower bound of entropic affinity root is not valid."
    assert (
        entropy_gap(end, C) > 0
    ).all(), "Lower bound of entropic affinity root is not valid."


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("optimizer", ["Adam", "LBFGS"])
@pytest.mark.parametrize("keops", lst_keops)
def test_sym_entropic_affinity(dtype, metric, optimizer, keops):
    n = 300
    X, _ = toy_dataset(n, dtype)
    perp = 30
    tol = 1e-2
    zeros = torch.zeros(n, dtype=getattr(torch, dtype), device=DEVICE)
    ones = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)
    lr = 1e0 if optimizer == "LBFGS" else 1e-1

    affinity = SymmetricEntropicAffinity(
        perplexity=perp,
        keops=keops,
        metric=metric,
        tol=1e-5,
        tolog=True,
        verbose=True,
        lr=lr,
        max_iter=3000,
        eps_square=True,
        device=DEVICE,
        optimizer=optimizer,
    )
    log_P = affinity(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, keops=keops)
    check_shape(log_P, (n, n))
    check_symmetry(log_P)
    check_marginal(log_P + math.log(n), zeros, dim=1, tol=tol, log=True)
    check_entropy(
        log_P + math.log(n), np.log(perp) * ones + 1, dim=1, tol=tol, log=True
    )


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", lst_keops)
def test_doubly_stochastic_entropic(dtype, metric, keops):
    n = 300
    X, _ = toy_dataset(n, dtype)
    eps = 1e0
    tol = 1e-3
    zeros = torch.zeros(n, dtype=getattr(torch, dtype), device=DEVICE)

    affinity = SinkhornAffinity(
        eps=eps,
        keops=keops,
        metric=metric,
        tol=tol,
        device=DEVICE,
        tolog=True,
        verbose=True,
    )
    log_P = affinity(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, keops=keops)
    check_shape(log_P, (n, n))
    check_symmetry(log_P)
    check_marginal(log_P + math.log(n), zeros, dim=1, tol=tol, log=True)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", lst_keops)
def test_doubly_stochastic_quadratic(dtype, metric, keops):
    n = 300
    X, _ = toy_dataset(n, dtype)
    eps = 1e0
    tol = 1e-3
    ones = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    affinity = DoublyStochasticQuadraticAffinity(
        eps=eps,
        keops=keops,
        metric=metric,
        tol=tol,
        device=DEVICE,
        tolog=True,
        verbose=True,
    )
    P = affinity(X)

    # -- check properties of the affinity matrix --
    check_type(P, keops=keops)
    check_shape(P, (n, n))
    check_symmetry(P)
    check_marginal(P, ones / n, dim=1, tol=tol, log=False)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("sparsity", [False])
@pytest.mark.parametrize("keops", lst_keops)
def test_umap_data_affinity(dtype, metric, sparsity, keops):
    n = 300
    X, _ = toy_dataset(n, dtype)
    n_neighbors = 30
    tol = 1e-3

    affinity = UMAPAffinityIn(
        n_neighbors=n_neighbors,
        device=DEVICE,
        keops=keops,
        metric=metric,
        tol=tol,
        verbose=True,
        sparsity=sparsity,
    )
    P = affinity(X)

    # -- check properties of the affinity matrix --
    check_type(P, keops=keops)
    check_shape(P, (n, n))
    check_nonnegativity(P)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", lst_keops)
@pytest.mark.parametrize("a, b", [(1, 2), (None, None)])
def test_umap_embedding_affinity(dtype, metric, keops, a, b):
    n = 300
    X, _ = toy_dataset(n, dtype)

    affinity = UMAPAffinityOut(
        device=DEVICE,
        keops=keops,
        metric=metric,
        verbose=True,
        a=a,
        b=b,
    )
    P = affinity(X)

    # -- check properties of the affinity matrix --
    check_type(P, keops=keops)
    check_shape(P, (n, n))
    check_nonnegativity(P)
    check_symmetry(P)
