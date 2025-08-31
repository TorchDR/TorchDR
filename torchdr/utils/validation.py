"""Useful functions for testing, compatible with KeOps."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from torch.testing import assert_close

try:
    import pandas as pd
except ImportError:
    pd = None

from .keops import is_lazy_tensor, pykeops


def check_NaNs(input, msg=None):
    """Check if a tensor contains NaN values."""
    if isinstance(input, list):
        for tensor in input:
            check_NaNs(tensor, msg)
    elif isinstance(input, torch.Tensor):
        if torch.isnan(input).any():
            raise ValueError(msg or "Tensor contains NaN values.")
    else:
        raise TypeError("Input must be a tensor or a list of tensors.")


def check_similarity_torch_keops(P, P_keops, K=None, test_indices=True, tol=1e-5):
    """Check that a torch.Tensor and a LazyTensor are equal on their largest entries."""
    assert isinstance(P, torch.Tensor), "P is not a torch.Tensor."

    assert P.shape == P_keops.shape, "P and P_keops do not have the same shape."

    n = P.shape[0]
    if K is None:
        K = n
    else:
        assert K <= n, "K is larger than the number of rows of P."

    if pykeops:
        assert is_lazy_tensor(P_keops), "P_keops is not a LazyTensor."
    else:
        assert_close(
            P,
            P_keops,
            atol=tol,
            rtol=tol,
            msg="Torch and Keops matrices are different.",
        )
        return

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
    # we only focus on values without duplicates for simplicity
    # notice that we cannot guarantee correspondence between the last topK
    if test_indices:
        try:
            assert_close(
                Largest_arg,
                Largest_keops_arg,
                atol=tol,
                rtol=tol,
                msg="Torch and Keops largest arguments are differentdue to duplicates.",
            )
        except AssertionError:
            for i in range(Largest_keops_values.shape[0]):
                Largest_values_unique, counts = torch.unique(
                    Largest_values[i, :], sorted=False, return_counts=True
                )
                # If the first test passes we know that these counts are the same
                # between torch and keops
                count_pos = 0
                pos = 0
                selected_args = []
                while count_pos < counts.shape[0] - 1:
                    if counts[count_pos] == 1:
                        selected_args.append(pos)

                    pos += counts[count_pos]
                    count_pos += 1

                assert_close(
                    Largest_arg[i, selected_args],
                    Largest_keops_arg[i, selected_args],
                    atol=tol,
                    rtol=tol,
                    msg=f"Torch and Keops largest arguments are different for row {i}.",
                )


def relative_similarity(P, P_target):
    """Compute similarity between a torch.Tensor or LazyTensor and a target."""
    return (P - P_target).abs().sum() / P_target.abs().sum()


def check_similarity(P, P_target, tol=1e-5, msg=None):
    """Check if a torch.Tensor or LazyTensor is close to a target matrix."""
    (n, p) = P.shape
    (also_n, also_p) = P_target.shape
    assert n == also_n and p == also_p, (
        "Matrix and target matrix do not have the same shape."
    )
    assert relative_similarity(P, P_target) < tol, (
        msg or "Matrix is not close to the target matrix."
    )


def check_symmetry(P, tol=1e-5, msg=None):
    """Check if a torch.Tensor or LazyTensor is symmetric."""
    check_similarity(P, P.T, tol=tol, msg=msg or "Matrix is not symmetric.")


def check_marginal(P, marg, dim=1, tol=1e-5, log=False):
    """Check if a torch.Tensor or LazyTensor has the correct marginal along axis dim.

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


def check_total_sum(P, total_sum, tol=1e-5):
    """Check if a torch.Tensor or LazyTensor has the correct total sum."""
    assert ((P.sum(0).sum() - total_sum) / total_sum).abs() < tol, (
        "Matrix has the wrong total sum."
    )


def check_entropy(P, entropy_target, dim=1, tol=1e-5, log=True):
    """Check if a torch.Tensor or LazyTensor has the correct entropy along axis dim."""
    from .utils import entropy

    assert_close(
        entropy(P, log=log, dim=dim),
        entropy_target,
        atol=tol,
        rtol=tol,
        msg=f"Matrix has the wrong entropy for dim={dim}.",
    )


def check_entropy_lower_bound(P, entropy_target, dim=1, log=True):
    """Check if a torch.Tensor or LazyTensor has the correct entropy along axis dim."""
    from .utils import entropy

    H = entropy(P, log=log, dim=dim)
    assert ((H - entropy_target) >= 0).all(), "Matrix entropy is lower than the target."


def check_type(P, keops):
    """Check if a tensor is a torch.Tensor or a LazyTensor (if keops is True)."""
    if keops:
        assert is_lazy_tensor(P), "Input is not a LazyTensor."
    else:
        assert isinstance(P, torch.Tensor), "Input is not a torch.Tensor."


def check_shape(P, shape):
    """Check if a tensor has the correct shape."""
    assert P.shape == shape, "Input shape is incorrect."


def check_nonnegativity(P):
    """Check if a tensor contains only non-negative values."""
    assert P.min() >= 0, "Input contains negative values."


def check_nonnegativity_eigenvalues(lambdas, tol_neg_ratio=1e-3, small_pos_ratio=1e-6):
    """Check if vector of eigenvalues lambdas contains only non-negative entries.

    This function checks if the input vector of eigenvalues contains significant
    negative entries, sets negative eigenvalues to zero if they are not significant,
    and removes eigenvalues that are too small.
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


def check_neighbor_param(n_neighbors, n_samples):
    r"""Check the n_neighbors parameter.

    If called with Tensors, do a clamp in‐graph so torch.compile can trace it."""

    # If either input is a tensor, switch to tensor‐only logic:
    if isinstance(n_neighbors, torch.Tensor) or isinstance(n_samples, torch.Tensor):
        # coerce both to long tensors on the same device
        n = torch.as_tensor(
            n_samples,
            dtype=torch.long,
            device=n_samples.device
            if isinstance(n_samples, torch.Tensor)
            else n_neighbors.device,
        )
        k = torch.as_tensor(n_neighbors, dtype=torch.long, device=n.device)

        # valid neighbor count is in [2, n−2]
        lower = torch.tensor(2, dtype=k.dtype, device=k.device)
        upper = n - 2

        return torch.clamp(k, min=lower, max=upper)

    if n_samples <= 1:
        raise ValueError(
            f"[TorchDR] ERROR : Input has less than one sample : n_samples = {n_samples}."
        )

    if n_neighbors <= 1 or n_neighbors >= n_samples - 1:
        raise ValueError(
            f"[TorchDR] ERROR : Number of requested neighbors must be greater than "
            f"1 and smaller than the number of samples - 1 (here {n_samples - 1}). "
            f"Got {n_neighbors}."
        )

    return n_neighbors


def validate_tensor(
    tensor,
    accept_sparse=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    ensure_2d=True,
    max_components=None,
):
    """Validate a torch tensor.

    This function validates an already-converted torch tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to validate.
    accept_sparse : bool, default=False
        Whether to accept sparse tensors.
    ensure_min_samples : int, default=1
        Make sure that the tensor has at least `ensure_min_samples` samples.
    ensure_min_features : int, default=1
        Make sure that the tensor has at least `ensure_min_features` features.
    ensure_2d : bool, default=True
        Whether to raise an error if the tensor is not 2D.
    max_components : int, optional
        If provided, ensures that n_features >= max_components.
        Useful for dimensionality reduction methods.

    Returns
    -------
    tensor_validated : torch.Tensor
        The validated tensor (may be reshaped if ensure_2d=True).
    """
    # Validate it's a tensor
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            "validate_tensor expects a torch.Tensor, got {}".format(type(tensor))
        )

    if torch.is_complex(tensor):
        raise ValueError("[TorchDR] ERROR : complex tensors are not supported.")

    if not torch.isfinite(tensor).all():
        raise ValueError("[TorchDR] ERROR : input contains infinite values.")

    if not accept_sparse and tensor.is_sparse:
        raise ValueError("Sparse tensors are not accepted.")

    if ensure_2d:
        if tensor.ndim == 0:
            raise ValueError("Expected 2D tensor, got scalar tensor instead.")
        elif tensor.ndim == 1:
            tensor = tensor.reshape(-1, 1)

        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {tensor.ndim}D tensor instead.")

    n_samples, n_features = tensor.shape
    if n_samples < ensure_min_samples:
        raise ValueError(
            f"Found tensor with {n_samples} samples, but a minimum of "
            f"{ensure_min_samples} is required."
        )

    if n_features < ensure_min_features:
        raise ValueError(
            f"Found tensor with {n_features} features, but a minimum of "
            f"{ensure_min_features} is required."
        )

    if max_components is not None and max_components > n_features:
        raise ValueError(
            f"n_components={max_components} is invalid for n_features={n_features}. "
            f"The number of components cannot exceed the number of features."
        )

    return tensor
