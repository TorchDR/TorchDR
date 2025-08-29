"""Distances based on pure PyTorch backend."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from torchdr.utils.utils import kmin


LIST_METRICS_TORCH = [
    "euclidean",
    "sqeuclidean",
    "manhattan",
    "angular",
    "sqhyperbolic",
]


def pairwise_distances_torch(
    X: torch.Tensor,
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    k: int = None,
    exclude_diag: bool = False,
    device: str = "auto",
):
    r"""Compute pairwise distances between points using PyTorch.

    When Y is not provided (i.e. computing distances within X) and
    `exclude_diag` is True, the self–distance for each point (i.e. the diagonal)
    is set to infinity so that the self index is not returned as a nearest neighbor.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        First dataset.
    Y : torch.Tensor of shape (m_samples, n_features), optional
        Second dataset. If None, Y is set to X.
    metric : str
        Metric to use for computing distances. Supported values are those in LIST_METRICS_KEOPS.
    k : int, optional
        Number of nearest neighbors to consider for the distances.
        If provided, the function returns a tuple (C, indices) where C contains the k smallest distances.
    exclude_diag : bool, default False
        If True and Y is not provided, the self–distance (diagonal elements) are set to infinity,
        excluding the self index from the k nearest neighbors.
    device : str, default="auto"
        Device to use for computation. If "auto", keeps data on its current device.
        Otherwise, temporarily moves data to specified device for computation.
        Output remains on the computation device.

    Returns
    -------
    C : torch.Tensor
        If k is None, C is the full pairwise distance matrix of shape (n_samples, m_samples).
        If k is provided, C is of shape (n_samples, k) containing the k smallest distances for each sample.
    indices : torch.Tensor or None
        If k is provided, indices is of shape (n_samples, k) containing the indices of the k nearest neighbors.
        Otherwise, None.
    """
    if metric not in LIST_METRICS_TORCH:
        raise ValueError(f"[TorchDR] ERROR : The '{metric}' distance is not supported.")

    # Move to computation device if needed
    # The moved tensors will be garbage collected after function returns
    if device != "auto" and str(X.device) != device:
        X = X.to(device)
        if Y is not None and Y is not X:
            Y = Y.to(device)

    # If Y is not provided, use X (and reuse its memory).
    if Y is None or Y is X:
        Y = X
        do_exclude_diag = exclude_diag
    else:
        do_exclude_diag = False  # Only exclude self when Y is not provided.

    # For metrics that require norms, compute once and reuse if Y is X.
    if metric in {"sqeuclidean", "euclidean", "sqhyperbolic"}:
        X_norm = (X**2).sum(dim=-1)
        if Y is X:
            Y_norm = X_norm
        else:
            Y_norm = (Y**2).sum(dim=-1)

    # Compute pairwise distances for each metric.
    if metric == "sqeuclidean":
        # (X_norm.unsqueeze(-1)) has shape (n, 1) and (Y_norm.unsqueeze(-2)) has shape (1, m).
        C = X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * (X @ Y.transpose(-1, -2))
    elif metric == "euclidean":
        C = X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * (X @ Y.transpose(-1, -2))
        C = C.clamp(min=0)
        C = C.sqrt()
    elif metric == "manhattan":
        # Note: This will create a large intermediate tensor with shape (n, m, d).
        C = (X.unsqueeze(-2) - Y.unsqueeze(-3)).abs().sum(dim=-1)
    elif metric == "angular":
        C = -(X @ Y.transpose(-1, -2))
    elif metric == "sqhyperbolic":
        denom = (1 - X_norm).unsqueeze(-1) * (1 - Y_norm).unsqueeze(-2)
        C = torch.relu(
            X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * X @ Y.transpose(-1, -2)
        )
        C = torch.arccosh(1 + 2 * (C / denom) + 1e-8) ** 2
    else:
        raise ValueError(f"[TorchDR] ERROR : Unsupported metric '{metric}'.")

    # If requested, exclude self–neighbors by setting the diagonal to a large number.
    if do_exclude_diag:
        n = C.shape[0]
        diag_idx = torch.arange(n, device=C.device)
        diag_mask = torch.zeros_like(C)
        diag_mask[diag_idx, diag_idx] = 1e12
        C = C + diag_mask

    # If k is provided, select the k smallest distances per row.
    if k is not None:
        C_knn, indices = kmin(C, k=k, dim=1)
        # Output stays on computation device for downstream use
        return C_knn, indices
    else:
        # Output stays on computation device for downstream use
        return C, None
