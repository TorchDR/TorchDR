# -*- coding: utf-8 -*-
"""
Various useful functions
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import functools
import itertools
import torch
from pykeops.torch import LazyTensor


def wrap_vectors(func):
    r"""
    Reshape all input vectors from size (n) to size (n, 1).
    If any input is a lazy tensor, convert all input vectors to lazy tensors.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        use_keops = any(
            isinstance(arg, LazyTensor)
            for arg in itertools.chain(args, kwargs.values())
        )
        is_vector = lambda arg: isinstance(arg, torch.Tensor) and arg.ndim == 1
        unsqueeze = lambda arg: (
            LazyTensor(arg[:, None], 0) if use_keops else arg[:, None]
        )

        args = [unsqueeze(arg) if is_vector(arg) else arg for arg in args]
        kwargs = {
            key: (unsqueeze(value) if is_vector(value) else value)
            for key, value in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapper


def kmin(A, k=1, dim=0):
    r"""
    Returns the k smallest element of a tensor or lazy tensor along axis dim.
    """
    if isinstance(A, LazyTensor):
        A_min = A.Kmin(K=k, dim=dim).squeeze()
        return A_min.T if dim == 0 else A_min
    else:
        return (
            A.topk(k=k, dim=dim, largest=False).values
            if k > 1
            else A.min(dim=dim).values
        )


def kmax(A, k=1, dim=0):
    r"""
    Returns the k largest element of a tensor or lazy tensor along axis dim.
    """
    if isinstance(A, LazyTensor):
        A_max = -(-A).Kmin(K=k, dim=dim).squeeze()
        return A_max.T if dim == 0 else A_max
    else:
        return (
            A.topk(k=k, dim=dim, largest=True).values
            if k > 1
            else A.max(dim=dim).values
        )


@wrap_vectors
def sum_matrix_vector(M, v):
    r"""
    Returns the sum of a matrix and a vector. M can be tensor or lazy tensor.
    """
    return M + v


def check_NaNs(input, msg=None):
    if isinstance(input, list):
        for tensor in input:
            check_NaNs(tensor, msg)
    elif isinstance(input, torch.Tensor):
        if torch.isnan(input).any():
            raise ValueError(msg or "Tensor contains NaN values.")
    else:
        raise TypeError("Input must be a tensor or a list of tensors.")
