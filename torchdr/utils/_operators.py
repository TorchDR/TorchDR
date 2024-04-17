# -*- coding: utf-8 -*-
"""
Useful common functions for defining objective functions and constraints
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

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
