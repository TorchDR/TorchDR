from torchdr.utils.root_finding import binary_search, false_position
import torch
import pytest

lst_types = [torch.double, torch.float]


@pytest.mark.parametrize("dtype", lst_types)
def test_binary_search(dtype):

    def f(x):
        return x**2 - 1

    # tes 1D
    begin = torch.tensor([-.5, ], dtype=dtype)
    end = torch.tensor([2.0, ], dtype=dtype)

    m = binary_search(
        f,
        1,
        begin=begin,
        end=end,
        max_iter=1000,
        tol=1e-9,
        verbose=False)
    assert torch.allclose(m, torch.tensor([1.], dtype=dtype), atol=1e-5)

    # test 2D
    begin = torch.tensor([-.5, -.5], dtype=dtype)
    end = torch.tensor([2.0, 2.0], dtype=dtype)
    m = binary_search(f, 2, begin=begin, end=end, max_iter=1000, tol=1e-9, verbose=True)
    assert torch.allclose(m, torch.tensor([1., 1.], dtype=dtype), atol=1e-5)


@pytest.mark.parametrize("dtype", lst_types)
def test_false_position(dtype):

    def f(x):
        return x**2 - 1

    # tes 1D
    begin = torch.tensor([-.5, ], dtype=dtype)
    end = torch.tensor([2.0, ], dtype=dtype)

    m = false_position(
        f,
        1,
        begin=begin,
        end=end,
        max_iter=1000,
        tol=1e-9,
        verbose=False)
    assert torch.allclose(m, torch.tensor([1.], dtype=dtype), atol=1e-5)

    # test 2D
    begin = torch.tensor([-.5, -.5], dtype=dtype)
    end = torch.tensor([2.0, 2.0], dtype=dtype)
    m = false_position(
        f,
        2,
        begin=begin,
        end=end,
        max_iter=1000,
        tol=1e-9,
        verbose=True)
    assert torch.allclose(m, torch.tensor([1., 1.], dtype=dtype), atol=1e-5)
