# -*- coding: utf-8 -*-
"""
Useful functions for defining objectives and constraints
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from pykeops.torch import LazyTensor

from torchdr.utils._wrappers import wrap_vectors, sum_all_axis


def entropy(P, log=True, dim=1):
    r"""
    Computes the entropy of P along axis dim.
    Supports log domain input.
    """
    if log:
        return -(P.exp() * (P - 1)).sum(dim).squeeze()
    else:
        return -(P * (P.log() - 1)).sum(dim).squeeze()


@sum_all_axis
def cross_entropy_loss(P, Q, log_Q=False):
    r"""
    Computes the cross-entropy between P and Q.
    Supports log domain input for Q.
    """
    if log_Q:
        return -P * Q
    else:
        return -P * Q.log()


@sum_all_axis
def square_loss(P, Q):
    r"""
    Computes the square loss between P and Q.
    """
    return (P - Q) ** 2


def kmin(A, k=1, dim=0):
    r"""
    Returns the k smallest elements of a tensor or lazy tensor along axis dim,
    along with corresponding indices.
    Output (both values and indices) of dim (n, k) if dim=1 and (k, n) if dim=0.
    """
    assert isinstance(dim, int), "dim should be an integer."
    if isinstance(A, LazyTensor):
        dim_red = lambda P: (
            P.T if dim == 0 else P
        )  # reduces the same axis as torch.topk
        values, indices = A.Kmin_argKmin(K=k, dim=dim)
        return dim_red(values), dim_red(indices)
    else:
        values, indices = A.topk(k=k, dim=dim, largest=False)
        return values, indices


def kmax(A, k=1, dim=0):
    r"""
    Returns the k largest elements of a tensor or lazy tensor along axis dim,
    along with corresponding indices.
    Output (both values and indices) of dim (n, k) if dim=1 and (k, n) if dim=0.
    """
    assert isinstance(dim, int), "dim should be an integer."
    if isinstance(A, LazyTensor):
        dim_red = lambda P: (
            P.T if dim == 0 else P
        )  # reduces the same axis as torch.topk
        values, indices = (-A).Kmin_argKmin(K=k, dim=dim)
        return -dim_red(values), dim_red(indices)
    else:
        values, indices = A.topk(k=k, dim=dim, largest=True)
        return values, indices


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
    If input is a lazy tensor, returns a lazy tensor that can be multiplied with P.
    """
    assert dim in [0, 1, (0, 1)]

    if isinstance(P, torch.Tensor):
        return P.sum(dim, keepdim=True)

    elif isinstance(P, LazyTensor):
        if dim == (0, 1):
            return P.sum(0).sum()  # shape (1)
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
            return log_P.logsumexp(1).logsumexp(0)  # shape (1)
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
    if dim is None:
        return P

    assert dim in [0, 1, (0, 1)]

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


@wrap_vectors
def sum_matrix_vector(M, v):
    r"""
    Returns the sum of a matrix and a vector. M can be tensor or lazy tensor.
    Equivalent to M + v[:, None].
    """
    return M + v


def identity_matrix(n, keops, device, dtype):
    r"""
    Returns the identity matrix of size n with corresponding device and dtype.
    Outputs a lazy tensor if keops is True.
    """
    if keops:  # (complicated) workaround for identity matrix with KeOps
        i = torch.arange(n).to(device=device, dtype=dtype)
        j = torch.arange(n).to(device=device, dtype=dtype)
        i = LazyTensor(i[:, None, None])
        j = LazyTensor(j[None, :, None])
        return (0.5 - (i - j) ** 2).step()
    else:
        return torch.eye(n, device=device, dtype=dtype)


def to_torch(x, device="cuda", verbose=True, return_backend_device=False):
    use_gpu = (device in ["cuda", "cuda:0", "gpu", None]) and torch.cuda.is_available()
    new_device = torch.device("cuda:0" if use_gpu else "cpu")

    if verbose:
        print(f"[TorchDR] Using device: {new_device}.")

    if isinstance(x, torch.Tensor):
        input_backend = "torch"
        input_device = x.device
        if input_device != new_device:
            x_ = x.to(new_device)
        else:
            x_ = x

    elif isinstance(x, np.ndarray):
        input_backend = "numpy"
        input_device = "cpu"
        x_ = torch.from_numpy(x).to(new_device)  # memory efficient

    else:
        raise ValueError(f"Unsupported type {type(x)}.")

    if return_backend_device:
        return x_, input_backend, input_device
    else:
        return x_
