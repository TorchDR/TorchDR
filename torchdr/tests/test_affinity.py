import torch
import numpy as np
import pytest

from torchdr.utils import (
    check_similarity_torch_keops,
    check_similarity,
    check_symmetry,
    check_marginal,
    check_entropy,
    check_type,
    check_shape,
    check_nonnegativity,
    check_total_sum,
    entropy,
    pairwise_distances,
)
from torchdr.affinity import (
    ScalarProductAffinity,
    GibbsAffinity,
    StudentAffinity,
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    DoublyStochasticEntropic,
    log_Pe,
    bounds_entropic_affinity,
)

lst_types = [torch.double, torch.float]

LIST_METRICS_TEST = ["euclidean", "manhattan"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("dtype", lst_types)
def test_scalar_product_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype, device=DEVICE)

    list_P = []
    for keops in [False, True]:
        affinity = ScalarProductAffinity(keops=keops)
        P = affinity.get(X)
        list_P.append(P)

        # -- check properties of the affinity matrix --
        check_type(P, keops=keops)
        check_shape(P, (n, n))
        check_symmetry(P)

    # --- check consistency between torch and keops ---
    check_similarity_torch_keops(list_P[0], list_P[1], K=10)


@pytest.mark.parametrize("dtype", lst_types)
def test_gibbs_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype, device=DEVICE)
    one = torch.ones(n, dtype=dtype, device=X.device)

    for metric in LIST_METRICS_TEST:
        for dim in [0, 1, (0, 1)]:
            list_P = []
            for keops in [False, True]:
                affinity = GibbsAffinity(keops=keops, metric=metric, dim=dim)
                P = affinity.get(X)
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
def test_student_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype, device=DEVICE)
    one = torch.ones(n, dtype=dtype, device=X.device)

    for metric in LIST_METRICS_TEST:
        for dim in [0, 1, (0, 1)]:
            list_P = []
            for keops in [False, True]:
                affinity = StudentAffinity(keops=keops, metric=metric, dim=dim)
                P = affinity.get(X)
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
def test_entropic_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype, device=DEVICE)
    perp = 5
    tol = 1e-5
    one = torch.ones(n, dtype=dtype, device=X.device)
    target_entropy = np.log(perp) * one + 1

    def entropy_gap(eps, C):  # function to find the root of
        return entropy(log_Pe(C, eps), log=True) - target_entropy

    for metric in LIST_METRICS_TEST:

        list_P = []
        for keops in [False, True]:
            affinity = EntropicAffinity(
                perplexity=perp, keops=keops, metric=metric, tol=tol, verbose=True
            )
            P = affinity.get(X)
            list_P.append(P)

            # -- check properties of the affinity matrix --
            check_type(P, keops=keops)
            check_shape(P, (n, n))
            check_marginal(P, one, dim=1)
            check_entropy(P, target_entropy, dim=1, tol=tol)

            # -- check bounds on the root of entropic affinities --
            C = pairwise_distances(X, metric=metric, keops=keops)
            begin, end = bounds_entropic_affinity(C, perplexity=perp)
            assert (
                entropy_gap(begin, C) < 0
            ).all(), "Lower bound of entropic affinity root is not valid"
            assert (
                entropy_gap(end, C) > 0
            ).all(), "Lower bound of entropic affinity root is not valid"

        # --- check consistency between torch and keops ---
        check_similarity_torch_keops(list_P[0], list_P[1], K=perp)


@pytest.mark.parametrize("dtype", lst_types)
def test_l2sym_entropic_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype, device=DEVICE)
    perp = 5

    for metric in LIST_METRICS_TEST:
        list_P = []
        for keops in [False, True]:
            affinity = L2SymmetricEntropicAffinity(
                perplexity=perp, keops=keops, metric=metric, verbose=True
            )
            P = affinity.get(X)
            list_P.append(P)

            # -- check properties of the affinity matrix --
            check_type(P, keops=keops)
            check_shape(P, (n, n))
            check_symmetry(P)

        # --- check consistency between torch and keops ---
        check_similarity_torch_keops(list_P[0], list_P[1], K=perp)


@pytest.mark.parametrize("dtype", lst_types)
def test_sym_entropic_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype, device=DEVICE)
    perp = 5
    tol = 1e-3
    one = torch.ones(n, dtype=dtype, device=X.device)
    target_entropy = np.log(perp) * one + 1

    for metric in LIST_METRICS_TEST:
        list_P = []
        for keops in [False, True]:
            affinity = SymmetricEntropicAffinity(
                perplexity=perp,
                keops=keops,
                metric=metric,
                tol=tol,
                tolog=True,
                verbose=True,
                lr=1e-1,
                max_iter=1000,
            )
            P = affinity.get(X)
            list_P.append(P)

            # -- check properties of the affinity matrix --
            check_type(P, keops=keops)
            check_shape(P, (n, n))
            check_symmetry(P)
            check_marginal(P, one, dim=1, tol=tol)
            check_entropy(P, target_entropy, dim=1, tol=tol)

            # --- test eps_square trick ---
            affinity_eps_square = SymmetricEntropicAffinity(
                perplexity=perp,
                keops=keops,
                metric=metric,
                tol=1e-5,
                eps_square=True,
                lr=1e-1,
                max_iter=1000,
            )
            P_eps_square = affinity_eps_square.get(X)
            check_similarity(P, P_eps_square, tol=tol)

        # --- check consistency between torch and keops ---
        check_similarity_torch_keops(list_P[0], list_P[1], K=perp)


@pytest.mark.parametrize("dtype", lst_types)
def test_doubly_stochastic_entropic(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype, device=DEVICE)
    eps = 1.0
    tol = 1e-3
    one = torch.ones(n, dtype=dtype, device=X.device)

    for metric in LIST_METRICS_TEST:
        list_P = []
        for keops in [False, True]:
            affinity = DoublyStochasticEntropic(eps=eps, keops=keops, metric=metric)
            P = affinity.get(X)
            list_P.append(P)

            # -- check properties of the affinity matrix --
            check_type(P, keops=keops)
            check_shape(P, (n, n))
            check_symmetry(P)
            check_marginal(P, one, dim=1, tol=tol)

        # --- check consistency between torch and keops ---
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)
