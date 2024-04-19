# -*- coding: utf-8 -*-
"""
Useful functions for defining objectives and constraints
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from pykeops.torch import LazyTensor


def entropy(P, log=True, dim=1):
    r"""
    Computes the entropy of P along axis dim.
    Supports log domain input.
    """
    if log:
        return -(P.exp() * (P - 1)).sum(dim).squeeze()
    else:
        return -(P * (P.log() - 1)).sum(dim).squeeze()


def cross_entropy_loss(P, Q, log_Q=False):
    r"""
    Computes the cross-entropy between P and Q.
    Supports log domain input for Q.
    """
    if log_Q:
        return -(P * Q).sum(1).sum()  # sum over both axis
    else:
        return -(P * Q.log()).sum(1).sum()


def square_loss(P, Q):
    r"""
    Computes the square loss between P and Q.
    """
    return ((P - Q) ** 2).sum(1).sum()


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


# inspired from svd_flip from sklearn.utils.extmath
def svd_flip(u, v):
    r"""
    Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


def _sum(P, dim):
    r"""
    Sums a 2d tensor along axis dim.
    If input is a torch tensor, returns a tensor with the same shape.
    If input is a lazy tensor, returns a lazy tensor that can be summed with P.
    """
    assert dim in [0, 1, (0, 1)]

    if isinstance(P, torch.Tensor):
        return P.sum(dim, keepdim=True)
    elif isinstance(P, LazyTensor):
        if dim == (0, 1):
            return P.sum(0).sum()  # scalar
        elif dim == 1:
            return P.sum(dim)[:, None]  # shape (n, 1, 1)
        elif dim == 0:
            return P.sum(dim)[None, :]  # shape (1, n, 1)
    else:
        raise ValueError("P should be a tensor or a lazy tensor.")


def _logsumexp(log_P, dim):
    r"""
    Logsumexp of a 2d tensor along axis dim.
    If input is a torch tensor, returns a tensor with the same shape.
    If input is a lazy tensor, returns a lazy tensor that can be summed with P.
    """
    assert dim in [0, 1, (0, 1)]

    if isinstance(log_P, torch.Tensor):
        return log_P.logsumexp(dim, keepdim=True)

    elif isinstance(log_P, LazyTensor):
        if dim == (0, 1):
            return log_P.logsumexp(0).logsumexp(0).squeeze()  # scalar
        elif dim == 1:
            return log_P.logsumexp(dim)[:, None]  # shape (n, 1, 1)
        elif dim == 0:
            return log_P.logsumexp(dim)[None, :]  # shape (1, n, 1)

    else:
        raise ValueError("log_P should be a tensor or a lazy tensor.")


def normalize_matrix(P, dim=1, log=False):
    r"""
    Normalizes a matrix along axis dim.
    If log, consider P in log domain and returns the normalized matrix in log domain.
    Handles both torch tensors and lazy tensors.
    """
    assert dim in [0, 1, (0, 1), None]

    if dim is None:
        return P

    if log:
        return P - _logsumexp(P, dim)
    else:
        return P / _sum(P, dim)


def center_kernel(K):
    r"""
    Centers a kernel matrix.
    """
    n = K.shape[0]
    K = K - _sum(K, dim=0) / n
    K = K - _sum(K, dim=1) / n
    K = K + _sum(K, dim=(0, 1)) / (n**2)
    return K
