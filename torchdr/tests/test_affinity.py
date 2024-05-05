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
from sklearn.datasets import make_moons

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
)
from torchdr.affinity import (
    ScalarProductAffinity,
    GibbsAffinity,
    StudentAffinity,
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    DoublyStochasticEntropic,
    DoublyStochasticQuadratic,
    UMAPAffinityData,
    UMAPAffinityEmbedding,
)
from torchdr.affinity.entropic import bounds_entropic_affinity, _log_Pe

lst_types = ["float32", "float64"]

LIST_METRICS_TEST = ["euclidean"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def toy_dataset(n=300, dtype="float32"):
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=0)
    return X.astype(dtype)


@pytest.mark.parametrize("dtype", lst_types)
def test_scalar_product_affinity(dtype):
    n = 300
    X = toy_dataset(n, dtype)

    list_P = []
    for keops in [False, True]:
        affinity = ScalarProductAffinity(device=DEVICE, keops=keops)
        P = affinity.fit_transform(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_symmetry(P)

    # --- check consistency between torch and keops ---
    check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_gibbs_affinity(dtype, metric, dim):
    n = 300
    X = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for keops in [False, True]:
        affinity = GibbsAffinity(
            device=DEVICE, keops=keops, metric=metric, dim_normalization=dim
        )
        P = affinity.fit_transform(X)
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
    check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("dim", [0, 1, (0, 1)])
def test_student_affinity(dtype, metric, dim):
    n = 300
    X = toy_dataset(n, dtype)
    one = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    list_P = []
    for keops in [False, True]:
        affinity = StudentAffinity(
            device=DEVICE, keops=keops, metric=metric, dim_normalization=dim
        )
        P = affinity.fit_transform(X)
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
    check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", [True, False])
@pytest.mark.parametrize("sparsity", [False, None])
def test_entropic_affinity(dtype, metric, keops, sparsity):
    n = 300
    X = toy_dataset(n, dtype)
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
    log_P = affinity.fit_transform(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, keops=keops)
    check_shape(log_P, (n, n))
    check_marginal(log_P, zeros, dim=1, tol=tol, log=True)
    check_entropy(log_P, target_entropy, dim=1, tol=tol, log=True)

    # -- check bounds on the root of entropic affinities --
    C = affinity._ground_cost_matrix(affinity.data_)
    begin, end = bounds_entropic_affinity(C, perplexity=perp)
    assert (
        entropy_gap(begin, C) < 0
    ).all(), "Lower bound of entropic affinity root is not valid."
    assert (
        entropy_gap(end, C) > 0
    ).all(), "Lower bound of entropic affinity root is not valid."


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", [True, False])
def test_l2sym_entropic_affinity(dtype, metric, keops):
    n = 300
    X = toy_dataset(n, dtype)
    perp = 30

    affinity = L2SymmetricEntropicAffinity(
        perplexity=perp, keops=keops, metric=metric, verbose=True, device=DEVICE
    )
    P = affinity.fit_transform(X)

    # -- check properties of the affinity matrix --
    check_type(P, keops=keops)
    check_shape(P, (n, n))
    check_symmetry(P)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("optimizer", ["Adam", "LBFGS"])
@pytest.mark.parametrize("keops", [True, False])
def test_sym_entropic_affinity(dtype, metric, optimizer, keops):
    n = 300
    X = toy_dataset(n, dtype)
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
    log_P = affinity.fit_transform(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, keops=keops)
    check_shape(log_P, (n, n))
    check_symmetry(log_P)
    check_marginal(log_P, zeros, dim=1, tol=tol, log=True)
    check_entropy(log_P, np.log(perp) * ones + 1, dim=1, tol=tol, log=True)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", [True, False])
def test_doubly_stochastic_entropic(dtype, metric, keops):
    n = 300
    X = toy_dataset(n, dtype)
    eps = 1e0
    tol = 1e-3
    zeros = torch.zeros(n, dtype=getattr(torch, dtype), device=DEVICE)

    affinity = DoublyStochasticEntropic(
        eps=eps,
        keops=keops,
        metric=metric,
        tol=tol,
        device=DEVICE,
        tolog=True,
        verbose=True,
    )
    log_P = affinity.fit_transform(X, log=True)

    # -- check properties of the affinity matrix --
    check_type(log_P, keops=keops)
    check_shape(log_P, (n, n))
    check_symmetry(log_P)
    check_marginal(log_P, zeros, dim=1, tol=tol, log=True)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", [True, False])
def test_doubly_stochastic_quadratic(dtype, metric, keops):
    n = 300
    X = toy_dataset(n, dtype)
    eps = 1e0
    tol = 1e-3
    ones = torch.ones(n, dtype=getattr(torch, dtype), device=DEVICE)

    affinity = DoublyStochasticQuadratic(
        eps=eps,
        keops=keops,
        metric=metric,
        tol=tol,
        device=DEVICE,
        tolog=True,
        verbose=True,
    )
    P = affinity.fit_transform(X)

    # -- check properties of the affinity matrix --
    check_type(P, keops=keops)
    check_shape(P, (n, n))
    check_symmetry(P)
    check_marginal(P, ones, dim=1, tol=tol, log=False)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", [True, False])
@pytest.mark.parametrize("sparsity", [None, False])
def test_umap_data_affinity(dtype, metric, keops, sparsity):
    n = 300
    X = toy_dataset(n, dtype)
    n_neighbors = 30
    tol = 1e-3

    affinity = UMAPAffinityData(
        n_neighbors=n_neighbors,
        device=DEVICE,
        keops=keops,
        metric=metric,
        tol=tol,
        verbose=True,
        sparsity=sparsity,
    )
    P = affinity.fit_transform(X)

    # -- check properties of the affinity matrix --
    check_type(P, keops=keops)
    check_shape(P, (n, n))
    check_nonnegativity(P)
    check_symmetry(P)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS_TEST)
@pytest.mark.parametrize("keops", [True, False])
@pytest.mark.parametrize("a, b", [(1, 2), (None, None)])
def test_umap_embedding_affinity(dtype, metric, keops, a, b):
    n = 300
    X = toy_dataset(n, dtype)

    affinity = UMAPAffinityEmbedding(
        device=DEVICE,
        keops=keops,
        metric=metric,
        verbose=True,
        a=a,
        b=b,
    )
    P = affinity.fit_transform(X)

    # -- check properties of the affinity matrix --
    check_type(P, keops=keops)
    check_shape(P, (n, n))
    check_nonnegativity(P)
    check_symmetry(P)
