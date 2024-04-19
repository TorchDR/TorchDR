import torch
import pytest

from torch.testing import assert_close
from torchdr.utils import check_equality_torch_keops, check_symmetry

from torchdr.utils import (
    pairwise_distances,
    binary_search,
    false_position,
    LIST_METRICS,
)

lst_types = [torch.double, torch.float]


@pytest.mark.parametrize("dtype", lst_types)
def test_binary_search(dtype):

    def f(x):
        return x**2 - 1

    # test 1D, with begin as scalar
    begin = 0.5
    end = None

    tol = 1e-9

    m = binary_search(
        f, 1, begin=begin, end=end, max_iter=1000, tol=tol, verbose=False, dtype=dtype
    )
    assert_close(m, torch.tensor([1.0], dtype=dtype))

    # test 2D, with begin as tensor
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

    # test 1D, with end as scalar
    begin = None
    end = 2

    tol = 1e-9

    m = false_position(
        f, 1, begin=begin, end=end, max_iter=1000, tol=tol, verbose=False, dtype=dtype
    )
    assert_close(m, torch.tensor([1.0], dtype=dtype))

    # test 2D, with end as tensor
    begin = None
    end = 2 * torch.ones(2, dtype=torch.int)

    m = false_position(
        f, 2, begin=begin, end=end, max_iter=1000, tol=1e-9, verbose=True, dtype=dtype
    )
    assert_close(m, torch.tensor([1.0, 1.0], dtype=dtype))


@pytest.mark.parametrize("dtype", lst_types)
def test_pairwise_distances(dtype):
    n, p = 100, 10
    x = torch.randn(n, p, dtype=dtype)

    for metric in LIST_METRICS:
        distances = pairwise_distances(x, metric=metric, keops=False)

        # check shape, symmetry
        assert distances.shape == (n, n)
        check_symmetry(
            distances,
            msg=f"Pairwise distance matrix with metric = {metric} if not symmetric.",
        )

        # check consistency with keops
        distances_keops = pairwise_distances(x, metric=metric, keops=True)
        check_equality_torch_keops(distances, distances_keops, K=10, tol=1e-3)
