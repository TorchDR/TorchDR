# -*- coding: utf-8 -*-
"""
Useful functions for defining objectives and constraints
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from pykeops.torch import LazyTensor

from torchdr.utils.wrappers import wrap_vectors


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
    Returns the k smallest elements of a tensor or lazy tensor along axis dim,
    along with corresponding indices.
    Output (both values and indices) of dim (n, k) if dim=1 and (k, n) if dim=0.
    """
    if not isinstance(dim, int):
        raise ValueError(
            "[TorchDR] ERROR : the input dim to kmin should be an integer."
        )

    if k >= A.shape[dim]:
        return A, torch.arange(A.shape[dim])

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
    if not isinstance(dim, int):
        raise ValueError(
            "[TorchDR] ERROR : the input dim to kmax should be an integer."
        )

    if k >= A.shape[dim]:
        return A, torch.arange(A.shape[dim])

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


def sum_red(P, dim):
    r"""
    Sums a 2d tensor along axis dim.
    If input is a torch tensor, returns a tensor with the same shape.
    If input is a lazy tensor, returns a lazy tensor that can be summed or
    multiplied with P.
    """
    ndim_input = len(P.shape)
    if ndim_input != 2:
        raise ValueError("[TorchDR] ERROR : input to sum_red should be a 2d tensor.")

    if dim is None:
        return 1

    if isinstance(P, torch.Tensor):
        return P.sum(dim, keepdim=True)

    elif isinstance(P, LazyTensor):
        if dim == (0, 1):
            return P.sum(1).sum(0)  # shape (1)
        elif dim == 1:
            return P.sum(dim)[:, None]  # shape (n, 1, 1)
        elif dim == 0:
            return P.sum(dim)[None, :]  # shape (1, n, 1)
        else:
            raise ValueError(
                f"[TorchDR] ERROR : invalid normalization_dim: {dim}. "
                "Should be (0, 1) or 0 or 1."
            )

    else:
        raise ValueError(
            "[TorchDR] ERROR : input to sum_red should be "
            "a torch.Tensor or a pykeops.torch.LazyTensor."
        )


def logsumexp_red(log_P, dim):
    r"""
    Logsumexp of a 2d tensor along axis dim.
    If input is a torch tensor, returns a tensor with the same shape.
    If input is a lazy tensor, returns a lazy tensor that can be summed
    or multiplied with P.
    """
    ndim_input = len(log_P.shape)
    if ndim_input != 2:
        raise ValueError(
            "[TorchDR] ERROR : input to logsumexp_red should be a 2d tensor."
        )

    if dim is None:
        return 0

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
            raise ValueError(
                f"[TorchDR] ERROR : invalid normalization_dim: {dim}. "
                "Should be (0, 1) or 0 or 1."
            )

    else:
        raise ValueError(
            "[TorchDR] ERROR : input to logsumexp_red should be "
            "a torch.Tensor or a pykeops.torch.LazyTensor."
        )


def normalize_matrix(P, dim=1, log=False):
    r"""
    Normalizes a matrix along axis dim.
    If log, consider P in log domain and returns the normalized matrix in log domain.
    Handles both torch tensors and lazy tensors.
    """
    if dim is None:
        return P

    if log:
        return P - logsumexp_red(P, dim)
    else:
        return P / sum_red(P, dim)


def extract_batch_normalization(normalization, indices, dim):
    r"""
    From a pre-computed normalization, extracts the normalization
    corresponding to batch indices.
    """
    if dim == (0, 1):
        return normalization  # normalization is a scalar so return as is
    elif dim == 0:
        return normalization[:, indices].transpose(0, 1)
    elif dim == 1:
        return normalization[indices]
    else:
        raise ValueError(
            f"[TorchDR] ERROR : invalid normalization_dim: {dim}."
            "Should be (0, 1) or 0 or 1."
        )


def center_kernel(K):
    r"""
    Centers a kernel matrix.
    """
    n = K.shape[0]
    K = K - sum_red(K, dim=0) / n
    K = K - sum_red(K, dim=1) / n
    K = K + sum_red(K, dim=(0, 1)) / (n**2)
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
    if keops:
        i = torch.arange(n).to(device=device, dtype=dtype)
        j = torch.arange(n).to(device=device, dtype=dtype)
        i = LazyTensor(i[:, None, None])
        j = LazyTensor(j[None, :, None])
        return (0.5 - (i - j) ** 2).step()
    else:
        return torch.eye(n, device=device, dtype=dtype)


def batch_transpose(arg):
    r"""
    Transposes a tensor or lazy tensor that can have a batch dimension.
    The batch dimension is the first, thus only the last two axis are transposed.
    """
    if isinstance(arg, LazyTensor):
        return arg.T
    elif isinstance(arg, torch.Tensor):
        return arg.transpose(-2, -1)
    else:
        raise ValueError(
            "[TorchDR] ERROR : Unsupported input shape for batch_transpose function."
        )
