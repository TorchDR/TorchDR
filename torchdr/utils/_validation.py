# -*- coding: utf-8 -*-
"""
Useful functions for testing
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from torch.testing import assert_close

from pykeops.torch import LazyTensor

from torchdr.utils._operators import entropy


def check_equality_torch_keops(P, P_keops, K=None, tol=1e-5):
    """
    Check that a torch.Tensor and a LazyTensor are equal on their largest entries.
    """
    assert isinstance(P, torch.Tensor), "P is not a torch.Tensor"
    assert isinstance(P_keops, LazyTensor), "P_keops is not a LazyTensor"
    assert P.shape == P_keops.shape, "P and P_keops do not have the same shape"

    n = P.shape[0]
    if K is None:
        K = n
    else:
        assert K <= n, "K is larger than the number of rows of P"

    # retrieve largest P_keops values and arguments
    Largest_keops_arg = (-P_keops).argKmin(K=K, dim=1)
    Largest_keops_values = -(-P_keops).Kmin(K=K, dim=1)

    # retrieve largest P values and arguments
    topk = P.topk(K, dim=1)
    Largest_arg = topk.indices
    Largest_values = topk.values

    # check that the largest values are the same
    assert_close(
        Largest_values,
        Largest_keops_values,
        atol=tol,
        rtol=tol,
        msg="Torch and Keops largest values are different.",
    )

    # check that the largest arguments are the same
    assert_close(
        Largest_arg,
        Largest_keops_arg,
        atol=tol,
        rtol=tol,
        msg="Torch and Keops largest arguments are different.",
    )


def check_symmetry(P, tol=1e-6):
    """
    Check if a torch.Tensor or LazyTensor is symmetric.
    """
    n = P.shape[0]
    also_n = P.shape[1]
    assert n == also_n, "Matrix is not square."
    assert (((P - P.T) ** 2).sum() / n**2) < tol, "Matrix is not symmetric."


def check_marginal(P, marg, dim=1, tol=1e-6):
    """
    Check if a torch.Tensor or LazyTensor has the correct marginal.
    """
    assert_close(
        P.sum(dim).squeeze(),
        marg,
        atol=tol,
        rtol=tol,
        msg="Matrix has the wrong marginal.",
    )


def check_entropy(P, entropy_target, dim=1, tol=1e-6):
    """
    Check if a torch.Tensor or LazyTensor has the correct entropy.
    """
    assert_close(
        entropy(P, log=False, dim=dim),
        entropy_target,
        atol=tol,
        rtol=tol,
        msg="Matrix has the wrong entropy",
    )


def check_NaNs(input, msg=None):
    """
    Check if a tensor contains NaN values.
    """
    if isinstance(input, list):
        for tensor in input:
            check_NaNs(tensor, msg)
    elif isinstance(input, torch.Tensor):
        if torch.isnan(input).any():
            raise ValueError(msg or "Tensor contains NaN values.")
    else:
        raise TypeError("Input must be a tensor or a list of tensors.")
