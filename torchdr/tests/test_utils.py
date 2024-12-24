"""Tests for functions in utils module."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch
from torch.testing import assert_close

from torchdr.utils import (
    LIST_METRICS,
    binary_search,
    center_kernel,
    check_shape,
    check_similarity,
    check_similarity_torch_keops,
    check_symmetry,
    false_position,
    handle_keops,
    pairwise_distances,
    pykeops,
    symmetric_pairwise_distances,
    symmetric_pairwise_distances_indices,
)

lst_types = [torch.double, torch.float]


# ====== test root finding methods ======


@pytest.mark.parametrize("dtype", lst_types)
def test_binary_search(dtype):
    def f(x):
        return x**2 - 1

    # --- test 1D, with begin as scalar ---
    begin = 0.5
    end = None

    tol = 1e-9

    m = binary_search(
        f, 1, begin=begin, end=end, max_iter=1000, tol=tol, verbose=False, dtype=dtype
    )
    assert_close(m, torch.tensor([1.0], dtype=dtype))

    # --- test 2D, with begin as tensor ---
    begin = 0.5 * torch.ones(2, dtype=torch.float16)
    end = None

    m = binary_search(
        f, 2, begin=begin, end=end, max_iter=1000, tol=tol, verbose=True, dtype=dtype
    )
    assert_close(m, torch.tensor([1.0, 1.0], dtype=dtype))


@pytest.mark.parametrize("dtype", lst_types)
def test_false_position(dtype):
    def f(x):
        return x**2 - 1

    # --- test 1D, with end as scalar ---
    begin = None
    end = 2

    tol = 1e-9

    m = false_position(
        f, 1, begin=begin, end=end, max_iter=1000, tol=tol, verbose=False, dtype=dtype
    )
    assert_close(m, torch.tensor([1.0], dtype=dtype))

    # --- test 2D, with end as tensor ---
    begin = None
    end = 2 * torch.ones(2, dtype=torch.int)

    m = false_position(
        f, 2, begin=begin, end=end, max_iter=1000, tol=1e-9, verbose=True, dtype=dtype
    )
    assert_close(m, torch.tensor([1.0, 1.0], dtype=dtype))


# ====== test pairwise distance matrices ======


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS)
def test_pairwise_distances(dtype, metric):
    n, m, p = 100, 50, 10
    x = torch.randn(n, p, dtype=dtype)
    y = torch.randn(m, p, dtype=dtype)

    # --- check consistency between torch and keops ---
    C = pairwise_distances(x, y, metric=metric, keops=False)
    check_shape(C, (n, m))


@pytest.mark.skipif(not pykeops, reason="pykeops is not available")
@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS)
def test_pairwise_distances_keops(dtype, metric):
    n, m, p = 100, 50, 10
    x = torch.randn(n, p, dtype=dtype)
    y = torch.randn(m, p, dtype=dtype)

    # --- check consistency between torch and keops ---
    C = pairwise_distances(x, y, metric=metric, keops=False)
    check_shape(C, (n, m))

    C_keops = pairwise_distances(x, y, metric=metric, keops=True)
    check_shape(C_keops, (n, m))

    check_similarity_torch_keops(C, C_keops, K=10)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS)
def test_symmetric_pairwise_distances(dtype, metric):
    n, p = 100, 10
    x = torch.randn(n, p, dtype=dtype)

    # --- check consistency between torch and keops ---
    C = symmetric_pairwise_distances(x, metric=metric, keops=False)
    check_shape(C, (n, n))
    check_symmetry(C)

    # --- check consistency with pairwise_distances ---
    C_ = pairwise_distances(x, metric=metric, keops=False)
    check_similarity(C, C_)


@pytest.mark.skipif(not pykeops, reason="pykeops is not available")
@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS)
def test_symmetric_pairwise_distances_keops(dtype, metric):
    n, p = 100, 10
    x = torch.randn(n, p, dtype=dtype)

    # --- check consistency between torch and keops ---
    C = symmetric_pairwise_distances(x, metric=metric, keops=False)
    check_shape(C, (n, n))
    check_symmetry(C)

    C_keops = symmetric_pairwise_distances(x, metric=metric, keops=True)
    check_shape(C_keops, (n, n))
    check_symmetry(C_keops)

    check_similarity_torch_keops(C, C_keops, K=10)

    # --- check consistency with pairwise_distances ---
    C_ = pairwise_distances(x, metric=metric, keops=False)
    check_similarity(C, C_)


@pytest.mark.parametrize("dtype", lst_types)
@pytest.mark.parametrize("metric", LIST_METRICS)
def test_symmetric_pairwise_distances_indices(dtype, metric):
    n, p = 100, 10
    x = torch.randn(n, p, dtype=dtype)
    indices = torch.randint(0, n, (n, 10))

    # --- check consistency with symmetric_pairwise_distances ---
    C_indices = symmetric_pairwise_distances_indices(x, indices, metric=metric)
    check_shape(C_indices, (n, 10))

    C_full = symmetric_pairwise_distances(x, metric=metric, keops=False)
    C_full_indices = C_full.gather(1, indices)

    check_similarity(C_indices, C_full_indices)


# ====== test center_kernel ======


@pytest.mark.parametrize("dtype", lst_types)
def test_center_kernel(dtype):
    torch.manual_seed(0)
    X = torch.randn(20, 30, dtype=dtype)
    K = X @ X.T
    K_c = center_kernel(K)
    n = K.shape[0]
    ones_n = torch.ones(n, dtype=dtype)
    H = torch.eye(n, dtype=dtype) - torch.outer(ones_n, ones_n) / n
    torch.testing.assert_close(K_c, H @ K @ H)


# ====== test handle_keops ======


class MockClass:
    def __init__(self, keops=False):
        self.keops = keops

    @handle_keops
    def some_method(self, *args, **kwargs):
        return "Function executed"


@pytest.fixture
def mock_obj():
    return MockClass()


def test_no_indices_keops_false(mock_obj):
    result = mock_obj.some_method()
    assert result == "Function executed"
    assert getattr(mock_obj, "keops_") is False  # Ensure keops_ remains False


def test_no_indices_keops_true(mock_obj):
    mock_obj.keops = True
    result = mock_obj.some_method()
    assert result == "Function executed"
    assert getattr(mock_obj, "keops_") is True  # Ensure keops_ remains True


def test_indices_provided(mock_obj):
    result = mock_obj.some_method(indices=[1, 2, 3])
    assert result == "Function executed"
    assert getattr(mock_obj, "keops_", None) is None  # Ensure keops_ isn't set
