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
    zeros = torch.zeros(n, dtype=dtype, device=X.device)
    ones = torch.ones(n, dtype=dtype, device=X.device)
    target_entropy = np.log(perp) * ones + 1

    def entropy_gap(eps, C):  # function to find the root of
        return entropy(log_Pe(C, eps), log=True) - target_entropy

    for metric in LIST_METRICS_TEST:

        list_P = []
        for keops in [False, True]:
            affinity = EntropicAffinity(
                perplexity=perp, keops=keops, metric=metric, tol=tol, verbose=True
            )
            log_P = affinity.get(X, log=True)
            list_P.append(log_P)

            # -- check properties of the affinity matrix --
            check_type(log_P, keops=keops)
            check_shape(log_P, (n, n))
            check_marginal(log_P, zeros, dim=1, tol=tol, log=True)
            check_entropy(log_P, target_entropy, dim=1, tol=1e-3, log=True)

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
    tol = 1e-2
    zeros = torch.zeros(n, dtype=dtype, device=X.device)
    ones = torch.ones(n, dtype=dtype, device=X.device)
    target_entropy = np.log(perp) * ones + 1

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
                lr=1e-3,
                max_iter=5000,
                eps_square=False,
            )
            log_P = affinity.get(X, log=True)
            list_P.append(log_P)

            # -- check properties of the affinity matrix --
            check_type(log_P, keops=keops)
            check_shape(log_P, (n, n))
            check_symmetry(log_P)
            check_marginal(log_P, zeros, dim=1, tol=tol, log=True)
            check_entropy(log_P, target_entropy, dim=1, tol=tol, log=True)

            # --- test eps_square trick ---
            affinity_eps_square = SymmetricEntropicAffinity(
                perplexity=perp,
                keops=keops,
                metric=metric,
                tol=tol,
                eps_square=True,
                lr=1e-3,
                max_iter=1000,
            )
            log_P_eps_square = affinity_eps_square.get(X, log=True)
            check_similarity(log_P, log_P_eps_square, tol=tol)

        # --- check consistency between torch and keops ---
        check_similarity_torch_keops(list_P[0], list_P[1], K=perp)


@pytest.mark.parametrize("dtype", lst_types)
def test_doubly_stochastic_entropic(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype, device=DEVICE)
    eps = 1e2
    tol = 1e-3
    zeros = torch.zeros(n, dtype=dtype, device=X.device)

    for metric in LIST_METRICS_TEST:
        list_P = []
        for keops in [False, True]:
            affinity = DoublyStochasticEntropic(
                eps=eps, keops=keops, metric=metric, tol=tol
            )
            log_P = affinity.get(X, log=True)
            list_P.append(log_P)

            # -- check properties of the affinity matrix --
            check_type(log_P, keops=keops)
            check_shape(log_P, (n, n))
            check_symmetry(log_P)
            check_marginal(log_P, zeros, dim=1, tol=tol, log=True)

        # --- check consistency between torch and keops ---
        check_similarity_torch_keops(list_P[0], list_P[1], K=10)
