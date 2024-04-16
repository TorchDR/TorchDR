import torch
import numpy as np
import pytest

from torch.testing import assert_close
from pykeops.torch import LazyTensor

from torchdr.tests.utils import check_equality_torch_keops
from torchdr.utils import pairwise_distances
from torchdr.affinity import (
    ScalarProductAffinity,
    GibbsAffinity,
    StudentAffinity,
    entropy,
    log_Pe,
    bounds_entropic_affinity,
    EntropicAffinity,
    L2SymmetricEntropicAffinity,
    SymmetricEntropicAffinity,
    DoublyStochasticEntropic,
)

lst_types = [torch.double, torch.float]

LIST_METRICS_TEST = ["euclidean", "manhattan"]


@pytest.mark.parametrize("dtype", lst_types)
def test_scalar_product_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype)

    # --- Without keops ---
    affinity = ScalarProductAffinity(keops=False)
    P = affinity.get(X)
    assert isinstance(P, torch.Tensor), "Affinity matrix is not a torch.Tensor"
    assert P.shape == (n, n), "Affinity matrix shape is incorrect"
    assert_close(P, P.T, msg="Affinity matrix is not symmetric")

    # --- With keops ---
    affinity_keops = ScalarProductAffinity(keops=True)
    P_keops = affinity_keops.get(X)
    assert isinstance(P_keops, LazyTensor), "Affinity matrix is not a LazyTensor"
    assert P_keops.shape == (n, n), "Affinity matrix shape is incorrect"
    assert ((P_keops - P_keops.T) ** 2).sum() < 1e-5, "Affinity matrix is not symmetric"

    # --- check equality between torch and keops ---
    check_equality_torch_keops(P, P_keops, K=10, tol=1e-5)


@pytest.mark.parametrize("dtype", lst_types)
def test_gibbs_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype)

    for metric in LIST_METRICS_TEST:
        # --- Without keops ---
        affinity = GibbsAffinity(keops=False, metric=metric)
        P = affinity.get(X)
        assert isinstance(P, torch.Tensor), "Affinity matrix is not a torch.Tensor"
        assert P.shape == (n, n), "Affinity matrix shape is incorrect"
        # assert_close(P, P.T, msg="Affinity matrix is not symmetric")
        assert P.min() >= 0, "Affinity matrix has negative values"

        # --- With keops ---
        affinity_keops = GibbsAffinity(keops=True, metric=metric)
        P_keops = affinity_keops.get(X)
        assert isinstance(P_keops, LazyTensor), "Affinity matrix is not a LazyTensor"
        assert P_keops.shape == (n, n), "Affinity matrix shape is incorrect"
        # assert (
        #     (P_keops - P_keops.T) ** 2
        # ).sum() < 1e-5, "Affinity matrix is not symmetric"
        assert P_keops.min() >= 0, "Affinity matrix has negative values"

        # --- check equality between torch and keops ---
        check_equality_torch_keops(P, P_keops, K=10, tol=1e-5)


@pytest.mark.parametrize("dtype", lst_types)
def test_student_affinity(dtype):
    n, p = 100, 10
    X = torch.randn(n, p, dtype=dtype)

    for metric in LIST_METRICS_TEST:
        # --- Without keops ---
        affinity = StudentAffinity(keops=False, metric=metric)
        P = affinity.get(X)
        assert isinstance(P, torch.Tensor), "Affinity matrix is not a torch.Tensor"
        assert P.shape == (n, n), "Affinity matrix shape is incorrect"
        # assert_close(P, P.T, msg="Affinity matrix is not symmetric")
        assert P.min() >= 0, "Affinity matrix has negative values"

        # --- With keops ---
        affinity_keops = StudentAffinity(keops=True, metric=metric)
        P_keops = affinity_keops.get(X)
        assert isinstance(P_keops, LazyTensor), "Affinity matrix is not a LazyTensor"
        assert P_keops.shape == (n, n), "Affinity matrix shape is incorrect"
        # assert (
        #     (P_keops - P_keops.T) ** 2
        # ).sum() < 1e-5, "Affinity matrix is not symmetric"
        assert P_keops.min() >= 0, "Affinity matrix has negative values"

        # --- check equality between torch and keops ---
        check_equality_torch_keops(P, P_keops, K=10, tol=1e-3)


@pytest.mark.parametrize("dtype", lst_types)
def test_entropic_affinity(dtype):
    n, p = 100, 10
    perp = 5
    target_entropy = np.log(perp) + 1
    tol = 1e-5

    def entropy_gap(eps, C):  # function to find the root of
        return entropy(log_Pe(C, eps), log=True) - target_entropy

    X = torch.randn(n, p, dtype=dtype)

    for metric in LIST_METRICS_TEST:

        # --- Without keops ---
        affinity_ea = EntropicAffinity(
            perplexity=perp, keops=False, metric=metric, tol=tol
        )
        P_ea = affinity_ea.get(X)
        assert isinstance(P_ea, torch.Tensor), "Affinity matrix is not a torch.Tensor"
        assert P_ea.shape == (n, n), "Affinity matrix shape is incorrect"
        assert_close(
            P_ea.sum(1),
            torch.ones(n, dtype=dtype),
            msg="Affinity matrix is not normalized row-wise",
        )
        H_ea = entropy(P_ea, log=False, dim=1)
        assert_close(
            H_ea - 1,
            np.log(perp) * torch.ones(n, dtype=dtype),
            atol=tol,
            rtol=tol,
            msg="Exp(Entropy-1) is not equal to the perplexity",
        )

        C = pairwise_distances(X, metric=metric, keops=False)
        begin, end = bounds_entropic_affinity(C, perplexity=perp)
        assert (
            entropy_gap(begin, C) < 0
        ).all(), "Lower bound of entropic affinity root is not valid"
        assert (
            entropy_gap(end, C) > 0
        ).all(), "Lower bound of entropic affinity root is not valid"

        # --- With keops ---
        affinity_ea_keops = EntropicAffinity(
            perplexity=perp, keops=True, metric=metric, tol=tol
        )
        P_ea_keops = affinity_ea_keops.get(X)
        assert isinstance(P_ea_keops, LazyTensor), "Affinity matrix is not a LazyTensor"
        assert P_ea_keops.shape == (n, n), "Affinity matrix shape is incorrect"
        assert_close(
            P_ea_keops.sum(1).squeeze(),
            torch.ones(n, dtype=dtype),
            msg="Affinity matrix is not normalized row-wise",
            atol=tol,
            rtol=tol,
        )
        H_ea_keops = entropy(P_ea_keops, log=False, dim=1)
        assert_close(
            H_ea_keops - 1,
            np.log(perp) * torch.ones(n, dtype=dtype),
            atol=tol,
            rtol=tol,
            msg="Exp(Entropy-1) is not equal to the perplexity",
        )

        C_keops = pairwise_distances(X, metric=metric, keops=True)
        begin_keops, end_keops = bounds_entropic_affinity(C_keops, perplexity=perp)
        assert (
            entropy_gap(begin_keops, C_keops) < 0
        ).all(), "Lower bound of entropic affinity root is not valid"
        assert (
            entropy_gap(end_keops, C_keops) > 0
        ).all(), "Lower bound of entropic affinity root is not valid"

        # --- check equality between torch and keops ---
        check_equality_torch_keops(P_ea, P_ea_keops, K=perp, tol=tol)


@pytest.mark.parametrize("dtype", lst_types)
def test_l2sym_entropic_affinity(dtype):
    n, p = 100, 10
    perp = 5
    tol = 1e-3

    X = torch.randn(n, p, dtype=dtype)

    for metric in LIST_METRICS_TEST:

        # --- Without keops ---
        affinity_l2ea = L2SymmetricEntropicAffinity(
            perplexity=perp, keops=False, metric=metric
        )
        P_l2ea = affinity_l2ea.get(X)
        assert isinstance(P_l2ea, torch.Tensor), "Affinity matrix is not a torch.Tensor"
        assert P_l2ea.shape == (n, n), "Affinity matrix shape is incorrect"
        assert_close(P_l2ea, P_l2ea.T, msg="Affinity matrix is not symmetric")

        # --- With keops ---
        affinity_l2ea_keops = L2SymmetricEntropicAffinity(
            perplexity=perp, keops=True, metric=metric
        )
        P_l2ea_keops = affinity_l2ea_keops.get(X)
        assert isinstance(
            P_l2ea_keops, LazyTensor
        ), "Affinity matrix is not a Lazy Tensor"
        assert P_l2ea_keops.shape == (n, n), "Affinity matrix shape is incorrect"
        assert (
            (P_l2ea_keops - P_l2ea_keops.T) ** 2
        ).sum() < tol, "Affinity matrix is not symmetric"

        # --- check equality between torch and keops ---
        check_equality_torch_keops(P_l2ea, P_l2ea_keops, K=perp, tol=tol)


@pytest.mark.parametrize("dtype", lst_types)
def test_sym_entropic_affinity(dtype):
    n, p = 100, 10
    perp = 5
    tol = 1e-3

    X = torch.randn(n, p, dtype=dtype)

    for metric in LIST_METRICS_TEST:

        # --- Without keops ---
        affinity_sea = SymmetricEntropicAffinity(
            perplexity=perp,
            keops=False,
            metric=metric,
            tol=1e-5,
            verbose=True,
            tolog=True,
        )
        P_sea = affinity_sea.get(X)
        assert isinstance(P_sea, torch.Tensor), "Affinity matrix is not a torch.Tensor"
        assert P_sea.shape == (n, n), "Affinity matrix shape is incorrect"
        assert_close(P_sea, P_sea.T, msg="Affinity matrix is not symmetric")
        assert_close(
            P_sea.sum(1),
            torch.ones(n, dtype=dtype),
            atol=tol,
            rtol=tol,
            msg="Affinity matrix is not normalized row-wise",
        )
        H_sea = entropy(P_sea, log=False, dim=1)
        assert_close(
            H_sea - 1,
            np.log(perp) * torch.ones(n, dtype=dtype),
            atol=tol,
            rtol=tol,
            msg="Exp(Entropy-1) is not equal to the perplexity",
        )

        # test eps_square
        affinity_sea_eps_square = SymmetricEntropicAffinity(
            perplexity=perp,
            keops=False,
            metric=metric,
            tol=1e-5,
            verbose=True,
            tolog=True,
            eps_square=True,
            lr=1e-1,
        )
        P_sea_eps_square = affinity_sea_eps_square.get(X)
        assert_close(
            P_sea,
            P_sea_eps_square,
            atol=tol,
            rtol=tol,
            msg="Eps square trick does not yield the same affinity matrix",
        )

        # --- With keops ---
        affinity_sea_keops = SymmetricEntropicAffinity(
            perplexity=perp,
            keops=True,
            metric=metric,
            tol=1e-5,
            verbose=True,
            tolog=True,
        )
        P_sea_keops = affinity_sea_keops.get(X)
        assert isinstance(
            P_sea_keops, LazyTensor
        ), "Affinity matrix is not a LazyTensor"
        assert P_sea_keops.shape == (n, n), "Affinity matrix shape is incorrect"
        assert (
            (P_sea_keops - P_sea_keops.T) ** 2
        ).sum() < 1e-5, "Affinity matrix is not symmetric"
        assert_close(
            P_sea_keops.sum(1).squeeze(),
            torch.ones(n, dtype=dtype),
            atol=tol,
            rtol=tol,
            msg="Affinity matrix is not normalized row-wise",
        )
        H_sea_keops = entropy(P_sea_keops, log=False, dim=1)
        assert_close(
            H_sea_keops - 1,
            np.log(perp) * torch.ones(n, dtype=dtype),
            atol=tol,
            rtol=tol,
            msg="Exp(Entropy-1) is not equal to the perplexity",
        )

        # --- check equality between torch and keops ---
        check_equality_torch_keops(P_sea, P_sea_keops, K=perp, tol=tol)


@pytest.mark.parametrize("dtype", lst_types)
def test_doubly_stochastic_entropic(dtype):
    n, p = 100, 10
    eps = 1.0
    tol = 1e-3

    X = torch.randn(n, p, dtype=dtype)

    for metric in LIST_METRICS_TEST:

        # --- Without keops ---
        affinity_ds = DoublyStochasticEntropic(eps=eps, keops=False, metric=metric)
        P_ds = affinity_ds.get(X)
        assert isinstance(P_ds, torch.Tensor), "Affinity matrix is not a torch.Tensor"
        assert P_ds.shape == (n, n), "Affinity matrix shape is incorrect"
        assert_close(P_ds, P_ds.T, msg="Affinity matrix is not symmetric")
        assert_close(
            P_ds.sum(1),
            torch.ones(n, dtype=dtype),
            atol=tol,
            rtol=tol,
            msg="Affinity matrix is not normalized row-wise",
        )

        # --- With keops ---
        affinity_ds_keops = DoublyStochasticEntropic(eps=eps, keops=True, metric=metric)
        P_ds_keops = affinity_ds_keops.get(X)
        assert isinstance(P_ds_keops, LazyTensor), "Affinity matrix is not a LazyTensor"
        assert P_ds_keops.shape == (n, n), "Affinity matrix shape is incorrect"
        assert (
            (P_ds_keops - P_ds_keops.T) ** 2
        ).sum() < tol, "Affinity matrix is not symmetric"
        assert_close(
            P_ds_keops.sum(1).squeeze(),
            torch.ones(n, dtype=dtype),
            atol=tol,
            rtol=tol,
            msg="Affinity matrix is not normalized row-wise",
        )

        # --- check equality between torch and keops ---
        check_equality_torch_keops(P_ds, P_ds_keops, K=10, tol=tol)
