# -*- coding: utf-8 -*-
"""
Useful wrappers for dealing with KeOps, vector dimensions etc...
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import functools
import itertools
import torch
from pykeops.torch import LazyTensor


def wrap_vectors(func):
    """
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


@wrap_vectors
def sum_matrix_vector(M, v):
    """
    Returns the sum of a matrix and a vector. M can be tensor or lazy tensor.
    """
    return M + v
