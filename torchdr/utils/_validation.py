# -*- coding: utf-8 -*-
"""
Useful functions for testing, compatible with KeOps
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from torch.testing import assert_close
from pykeops.torch import LazyTensor

from torchdr.utils._operators import entropy


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


def check_similarity_torch_keops(P, P_keops, K=None, tol=1e-3):
    """
    Check that a torch.Tensor and a LazyTensor are equal on their largest entries.
    """
    assert isinstance(P, torch.Tensor), "P is not a torch.Tensor."
    assert isinstance(P_keops, LazyTensor), "P_keops is not a LazyTensor."
    assert P.shape == P_keops.shape, "P and P_keops do not have the same shape."

    n = P.shape[0]
    if K is None:
        K = n
    else:
        assert K <= n, "K is larger than the number of rows of P."

    # retrieve largest P_keops values and arguments
    Largest_keops_values, Largest_keops_arg = (-P_keops).Kmin_argKmin(K=K, dim=1)
    Largest_keops_values = -Largest_keops_values

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


def relative_similarity(P, P_target):
    """
    Computes similarity between a torch.Tensor or LazyTensor and a target.
    """
    return (P - P_target).abs().sum() / P_target.abs().sum()


def check_similarity(P, P_target, tol=1e-6, msg=None):
    """
    Check if a torch.Tensor or LazyTensor is close to a target matrix.
    """
    (n, p) = P.shape
    (also_n, also_p) = P_target.shape
    assert (
        n == also_n and p == also_p
    ), "Matrix and target matrix do not have the same shape."
    assert relative_similarity(P, P_target) < tol, (
        msg or "Matrix is not close to the target matrix."
    )


def check_symmetry(P, tol=1e-6, msg=None):
    """
    Check if a torch.Tensor or LazyTensor is symmetric.
    """
    check_similarity(P, P.T, tol=tol, msg=msg or "Matrix is not symmetric.")


def check_marginal(P, marg, dim=1, tol=1e-6, log=False):
    """
    Check if a torch.Tensor or LazyTensor has the correct marginal along axis dim.
    If log is True, considers that both P and marg are in log domain.
    """
    if log:
        P_sum = P.logsumexp(dim).squeeze()
    else:
        P_sum = P.sum(dim).squeeze()

    assert_close(
        P_sum,
        marg,
        atol=tol,
        rtol=tol,
        msg=f"Matrix has the wrong marginal for dim={dim}.",
    )


def check_total_sum(P, total_sum, tol=1e-6):
    """
    Check if a torch.Tensor or LazyTensor has the correct total sum.
    """
    assert (
        (P.sum(0).sum() - total_sum) / total_sum
    ).abs() < tol, "Matrix has the wrong total sum."


def check_entropy(P, entropy_target, dim=1, tol=1e-6, log=True):
    """
    Check if a torch.Tensor or LazyTensor has the correct entropy along axis dim.
    """
    assert_close(
        entropy(P, log=log, dim=dim),
        entropy_target,
        atol=tol,
        rtol=tol,
        msg=f"Matrix has the wrong entropy for dim={dim}.",
    )


def check_entropy_lower_bound(P, entropy_target, dim=1, log=True):
    """
    Check if a torch.Tensor or LazyTensor has the correct entropy along axis dim.
    """
    H = entropy(P, log=log, dim=dim)
    assert ((H - entropy_target) >= 0).all(), "Matrix entropy is lower than the target."


def check_type(P, keops):
    """
    Check if a tensor is a torch.Tensor or a LazyTensor (if keops is True).
    """
    if keops:
        assert isinstance(P, LazyTensor), "Input is not a LazyTensor."
    else:
        assert isinstance(P, torch.Tensor), "Input is not a torch.Tensor."


def check_shape(P, shape):
    """
    Check if a tensor has the correct shape.
    """
    assert P.shape == shape, "Input shape is incorrect."


def check_nonnegativity(P):
    """
    Check if a tensor contains only non-negative values.
    """
    assert P.min() >= 0, "Input contains negative values."


def check_nonnegativity_eigenvalues(lambdas, tol_neg_ratio=1e-3, small_pos_ratio=1e-6):
    """
    Check if vector of eigenvalues lambdas contains only non-negative entries.
    """
    max_lam = lambdas.max()
    min_lam = lambdas.min()

    # check if negative eigenvalues are significant
    if min_lam < -tol_neg_ratio * max_lam:
        raise ValueError("Input matrix has significant negative eigenvalues.")

    # set negative eigenvalues to zero
    elif min_lam < 0:
        lambdas[lambdas < 0] = 0

    # remove eigenvalues that are too small
    too_small_lambdas = (0 < lambdas) & (lambdas < small_pos_ratio * max_lam)
    if too_small_lambdas.any():
        lambdas[too_small_lambdas] = 0

    return lambdas
