"""Useful functions for defining objectives and constraints."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause License

import os
import random
import time
import logging
import numpy as np
import torch
from typing import Union
import warnings

from .keops import is_lazy_tensor, LazyTensor, LazyTensorType
from .wrappers import wrap_vectors


def set_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Set up a logger for a given name.

    Parameters
    ----------
    name : str
        The name of the logger.
    verbose : bool, optional
        Whether to set the logger level to INFO (if True) or WARNING (if False).
        Default is False.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[TorchDR] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    return logger


def seed_everything(seed, fast=True, deterministic=False):
    """Seed all random number generators for reproducibility.

    Sets the seed for Python's random module, NumPy, PyTorch (CPU and GPU),
    and environment variables to ensure reproducible results across different runs.

    Parameters
    ----------
    seed : int or None
        The seed value to use. If None or negative, uses current time as seed.
    fast : bool, optional (default=True)
        If True, enables fast but non-deterministic cuDNN operations.
        If False, ensures deterministic cuDNN operations but may be slower.
    deterministic : bool, optional (default=False)
        If True, enables torch.use_deterministic_algorithms for maximum reproducibility.
        This may significantly slow down training but ensures deterministic behavior.

    Returns
    -------
    int
        The actual seed value used.
    """
    if seed is None or not isinstance(seed, int) or seed < 0:
        seed = int(time.time())
    else:
        seed = int(seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if fast:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if deterministic:
        torch.use_deterministic_algorithms(True)
        # Set environment variable for CUBLAS workspace configuration
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return seed


def cross_entropy_loss(P, Q, log=False):
    r"""Compute the cross-entropy loss between two probability distributions.

    Computes the cross-entropy H(P, Q) = -sum(P * log(Q)).
    Supports both regular and log-domain inputs for Q.

    Parameters
    ----------
    P : torch.Tensor or LazyTensor
        Source probability distribution of shape ``(n, m)``.
    Q : torch.Tensor or LazyTensor
        Target probability distribution of shape ``(n, m)``.
        If ``log=True``, Q should contain log-probabilities.
    log : bool, optional (default=False)
        If True, Q contains log-probabilities. If False, Q contains probabilities.

    Returns
    -------
    torch.Tensor or LazyTensor
        The cross-entropy loss value.
    """
    if log:
        return -sum_red(P * Q, dim=(0, 1))
    else:
        return -sum_red(P * Q.log(), dim=(0, 1))


def square_loss(P, Q):
    r"""Compute the squared Euclidean loss between two tensors.

    Computes the element-wise squared differences and sums them.

    Parameters
    ----------
    P : torch.Tensor or LazyTensor
        First tensor of shape ``(n, m)``.
    Q : torch.Tensor or LazyTensor
        Second tensor of shape ``(n, m)``.

    Returns
    -------
    torch.Tensor or LazyTensor
        The squared loss value.
    """
    return sum_red((P - Q) ** 2, dim=(0, 1))


def entropy(P, log=True, dim=1):
    r"""Compute the Shannon entropy of a probability distribution.

    Computes H(P) = -sum(P * log(P)) along the specified dimension.
    Supports both regular and log-domain inputs.

    Parameters
    ----------
    P : torch.Tensor or LazyTensor
        Probability distribution. If ``log=True``, contains log-probabilities.
    log : bool, optional (default=True)
        If True, P contains log-probabilities. If False, P contains probabilities.
    dim : int, optional (default=1)
        Dimension along which to compute the entropy.

    Returns
    -------
    torch.Tensor or LazyTensor
        The entropy values with the specified dimension reduced.
    """
    if log:
        return -(P.exp() * (P - 1)).sum(dim).squeeze()
    else:
        return -(P * (P.log() - 1)).sum(dim).squeeze()


def kmin(A, k=1, dim=0):
    r"""Return the k smallest elements and corresponding indices along axis dim.

    Output (both values and indices) of dim (n, k) if dim=1 and (k, n) if dim=0.

    Parameters
    ----------
    A : torch.Tensor or LazyTensor
        Input tensor of shape ``(n, m)``.
    k : int, optional (default=1)
        Number of smallest elements to return.
    dim : int, optional (default=0)
        Dimension along which to find the smallest elements.

    Returns
    -------
    tuple of (torch.Tensor or LazyTensor, torch.Tensor or None)
        - **values**: The k smallest values.
        - **indices**: The indices of the k smallest values, or None if k >= A.shape[dim].

    Raises
    ------
    ValueError
        If dim is not an integer.
    """
    if not isinstance(dim, int):
        raise ValueError(
            "[TorchDR] ERROR : the input dim to kmin should be an integer."
        )

    if k >= A.shape[dim]:
        return A, None

    if is_lazy_tensor(A):

        def dim_red(P):
            return P.T if dim == 0 else P  # reduces the same axis as torch.topk

        values, indices = A.Kmin_argKmin(K=k, dim=dim)
        return dim_red(values), dim_red(indices).int()

    else:
        values, indices = A.topk(k=k, dim=dim, largest=False)
        return values, indices.int()


def kmax(A, k=1, dim=0):
    r"""Return the k largest elements and corresponding indices along axis dim.

    Output (both values and indices) of dim (n, k) if dim=1 and (k, n) if dim=0.

    Parameters
    ----------
    A : torch.Tensor or LazyTensor
        Input tensor of shape ``(n, m)``.
    k : int, optional (default=1)
        Number of largest elements to return.
    dim : int, optional (default=0)
        Dimension along which to find the largest elements.

    Returns
    -------
    tuple of (torch.Tensor or LazyTensor, torch.Tensor)
        - **values**: The k largest values.
        - **indices**: The indices of the k largest values.

    Raises
    ------
    ValueError
        If dim is not an integer.
    """
    if not isinstance(dim, int):
        raise ValueError(
            "[TorchDR] ERROR : the input dim to kmax should be an integer."
        )

    if k >= A.shape[dim]:
        return A, torch.arange(A.shape[dim]).int()

    if is_lazy_tensor(A):

        def dim_red(P):
            return P.T if dim == 0 else P  # reduces the same axis as torch.topk

        values, indices = (-A).Kmin_argKmin(K=k, dim=dim)
        return -dim_red(values), dim_red(indices).int()

    else:
        values, indices = A.topk(k=k, dim=dim, largest=True)
        return values, indices.int()


def svd_flip(u, v, u_based_decision=True):
    r"""Sign correction to ensure deterministic output from SVD.

    Adjust the columns of ``u`` and the rows of ``v`` such that the loadings
    that are largest in absolute value in either ``u`` or ``v`` are always
    made positive, depending on the ``u_based_decision`` parameter.

    Parameters
    ----------
    u : torch.Tensor
        Left singular vectors of shape ``(n_samples, n_components)``
        (or a shape compatible with your SVD usage).
    v : torch.Tensor
        Right singular vectors of shape ``(n_components, n_features)``
        (or a shape compatible with your SVD usage).
    u_based_decision : bool, optional (default=True)
        - If ``True``, the signs are determined by examining the largest
          absolute values in each column of ``u``.
        - If ``False``, the signs are determined by examining the largest
          absolute values in each row of ``v``.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        - **u_flipped**: The sign-corrected version of ``u``.
        - **v_flipped**: The sign-corrected version of ``v``.
    """
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
    else:
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
    u *= signs[: u.shape[1]].view(1, -1)
    v *= signs.view(-1, 1)
    return u, v


def sum_red(P, dim):
    r"""Sum a 2d tensor along axis dim.

    If input is a torch tensor, return a tensor with the same shape.
    If input is a lazy tensor, return a lazy tensor that can be summed or
    multiplied with P.

    Parameters
    ----------
    P : torch.Tensor or LazyTensor
        Input 2D tensor to sum.
    dim : int or tuple of int or None
        Dimension(s) along which to sum. Can be 0, 1, (0, 1), or None.

    Returns
    -------
    torch.Tensor or LazyTensor
        Summed tensor with appropriate shape based on the input type and dimension.

    Raises
    ------
    ValueError
        If input is not a 2D tensor or if dim is invalid.
    """
    ndim_input = len(P.shape)
    if ndim_input != 2:
        raise ValueError("[TorchDR] ERROR : input to sum_red should be a 2d tensor.")

    if dim is None:
        return 1

    if isinstance(P, torch.Tensor):
        return P.sum(dim, keepdim=True)

    elif is_lazy_tensor(P):
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
    r"""Logsumexp of a 2d tensor along axis dim.

    If input is a torch tensor, return a tensor with the same shape.
    If input is a lazy tensor, return a lazy tensor that can be summed
    or multiplied with P.

    Parameters
    ----------
    log_P : torch.Tensor or LazyTensor
        Input 2D tensor containing log-probabilities.
    dim : int or tuple of int or None
        Dimension(s) along which to compute logsumexp. Can be 0, 1, (0, 1), or None.

    Returns
    -------
    torch.Tensor or LazyTensor
        Logsumexp result with appropriate shape based on the input type and dimension.

    Raises
    ------
    ValueError
        If input is not a 2D tensor or if dim is invalid.
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

    elif is_lazy_tensor(log_P):
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


def center_kernel(K, return_all=False):
    r"""Center a kernel matrix by removing row and column means.

    Applies double centering to a kernel matrix: K_centered = K - row_means - col_means + global_mean.
    This operation is commonly used in kernel PCA and other kernel methods.

    Parameters
    ----------
    K : torch.Tensor or LazyTensor
        Kernel matrix of shape ``(n, n)``.
    return_all : bool, optional (default=False)
        If True, returns all intermediate values (row means, column means, global mean).
        If False, returns only the centered kernel matrix.

    Returns
    -------
    torch.Tensor or LazyTensor or tuple
        If ``return_all=False``: The centered kernel matrix.
        If ``return_all=True``: A tuple ``(K_centered, row_mean, col_mean, global_mean)``.
    """
    n, d = K.shape
    row_mean = sum_red(K, dim=1) / d
    col_mean = sum_red(K, dim=0) / n
    mean = col_mean.mean()
    K = K - row_mean - col_mean + mean
    if return_all:
        return K, row_mean, col_mean, mean
    return K


@wrap_vectors
def sum_matrix_vector(M, v, transpose=False):
    r"""Return the sum of a matrix and a vector.

    M can be tensor or lazy tensor.
    Equivalent to `M + v[:, None]` if `transpose=False` else `M + v[None, :]`.

    Parameters
    ----------
    M : torch.Tensor or LazyTensor
        Input matrix.
    v : torch.Tensor
        Input vector to add to the matrix.
    transpose : bool, optional (default=False)
        If False, adds v as column vector. If True, adds v as row vector.

    Returns
    -------
    torch.Tensor or LazyTensor
        The sum of the matrix and vector.
    """
    if transpose:
        v = matrix_transpose(v)
    return M + v


@wrap_vectors
def prod_matrix_vector(M, v, transpose=False):
    r"""Return the product of a matrix and a vector.

    M can be tensor or lazy tensor.
    Equivalent to `M * v[:, None]` if `transpose=False` else `M * v[None, :]`.

    Parameters
    ----------
    M : torch.Tensor or LazyTensor
        Input matrix.
    v : torch.Tensor
        Input vector to multiply with the matrix.
    transpose : bool, optional (default=False)
        If False, multiplies v as column vector. If True, multiplies v as row vector.

    Returns
    -------
    torch.Tensor or LazyTensor
        The element-wise product of the matrix and vector.
    """
    if transpose:
        v = matrix_transpose(v)
    return M * v


def identity_matrix(n, keops, device, dtype):
    r"""Return the identity matrix of size n with corresponding device and dtype.

    Output a lazy tensor if keops is True.

    Parameters
    ----------
    n : int
        Size of the identity matrix (n x n).
    keops : bool
        If True, returns a KeOps LazyTensor. If False, returns a torch.Tensor.
    device : torch.device
        Device on which to create the matrix.
    dtype : torch.dtype
        Data type of the matrix elements.

    Returns
    -------
    torch.Tensor or LazyTensor
        Identity matrix of shape ``(n, n)``.
    """
    if keops:
        i = torch.arange(n).to(device=device, dtype=dtype)
        j = torch.arange(n).to(device=device, dtype=dtype)
        i = LazyTensor(i[:, None, None])
        j = LazyTensor(j[None, :, None])
        return (0.5 - (i - j) ** 2).step()
    else:
        return torch.eye(n, device=device, dtype=dtype)


def matrix_transpose(arg):
    r"""Transpose a tensor or lazy tensor matrix that can have a batch dimension.

    The batch dimension is the first, thus only the last two axis are transposed.

    Parameters
    ----------
    arg : torch.Tensor or LazyTensor
        Input tensor to transpose. Can have batch dimensions.

    Returns
    -------
    torch.Tensor or LazyTensor
        Transposed tensor with the last two dimensions swapped.

    Raises
    ------
    ValueError
        If input type is not supported.
    """
    if is_lazy_tensor(arg):
        return arg.T
    elif isinstance(arg, torch.Tensor):
        return arg.transpose(-2, -1)
    else:
        raise ValueError(
            f"[TorchDR] ERROR : Unsupported input type for matrix_transpose function: {type(arg)}."
        )


def bool_arg(arg):
    """Convert various argument types to boolean values.

    Handles conversion of different argument types to boolean values.
    For arrays and lists, returns True if any element is True.

    Parameters
    ----------
    arg : bool, int, float, list, np.ndarray, or array-like
        The argument to convert to boolean.

    Returns
    -------
    bool
        The boolean representation of the argument.
        For arrays/lists: True if any element is truthy, False otherwise.
        For scalars: the boolean value of the argument.
    """
    if isinstance(arg, (list, np.ndarray)):
        return bool(np.asarray(arg).any())
    else:
        return bool(arg)


def matrix_power(matrix: Union[torch.Tensor, LazyTensorType], power: float):
    r"""Compute the matrix power A^p for symmetric positive definite matrices.

    Supports both integer and non-integer powers for torch tensors.
    For KeOps lazy tensors, only integer powers are supported.

    For non-integer powers, uses eigendecomposition: A^p = Q * diag(λ^p) * Q^T,
    where Q contains the eigenvectors and λ the eigenvalues of A.

    Parameters
    ----------
    matrix : torch.Tensor or LazyTensor
        Input matrix of shape ``(n, n)`` or ``(..., n, n)``.
        Should be symmetric positive definite for non-integer powers.
    power : float
        The power to raise the matrix to. Must be non-negative.

    Returns
    -------
    torch.Tensor or LazyTensor
        The matrix raised to the specified power, same shape as input.

    Raises
    ------
    ValueError
        If power is negative.
    NotImplementedError
        If non-integer power is used with KeOps backend.

    Notes
    -----
    - For power=0, returns the identity matrix.
    - For power=1, returns the original matrix.
    - For integer powers > 1, uses repeated multiplication (KeOps) or
      torch.linalg.matrix_power (torch tensors).
    - For non-integer powers, uses eigendecomposition and requires the matrix
      to be symmetric positive definite.
    """
    if power < 0:
        raise ValueError("[TorchDR] ERROR: Negative matrix powers are not supported.")

    if is_lazy_tensor(matrix):
        raise NotImplementedError(
            "[TorchDR] ERROR: matrix powers are not supported with KeOps backend."
        )
    else:
        if power == int(power):
            power = int(power)
            if power == 0:
                n = matrix.shape[-1]
                return identity_matrix(
                    n, keops=False, device=matrix.device, dtype=matrix.dtype
                )
            elif power == 1:
                return matrix
            else:
                return torch.linalg.matrix_power(matrix, power)
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            eigenvalues = torch.clamp(eigenvalues, min=1e-12)
            powered_eigenvalues = eigenvalues**power
            return (
                eigenvectors
                @ torch.diag_embed(powered_eigenvalues)
                @ eigenvectors.transpose(-2, -1)
            )


def compile_func(func):
    """Compile a function with torch.compile if requested on the object."""
    try:
        return torch.compile(func)
    except Exception as e:
        warnings.warn(
            f"[TorchDR] WARNING: Could not compile {func.__name__} with torch.compile. "
            f"Falling back to eager execution. Reason: {e}",
            UserWarning,
        )
        return func
