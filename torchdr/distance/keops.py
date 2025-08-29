"""Distances based on KeOps backend."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from torchdr.utils.utils import identity_matrix, kmin

from torchdr.utils.keops import LazyTensor

LIST_METRICS_KEOPS = ["euclidean", "sqeuclidean", "manhattan", "angular"]


@torch.compiler.disable
def pairwise_distances_keops(
    X: torch.Tensor,
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    k: int = None,
    exclude_diag: bool = False,
    device: str = "auto",
):
    r"""Compute pairwise distances between points using KeOps LazyTensors.

    When Y is not provided (i.e. computing distances within X) and
    `exclude_diag` is True, the self–distance for each point (diagonal)
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
        If True and Y is not provided, the self–distance (diagonal entries) are set to infinity,
        excluding the self index from the k nearest neighbors.
    device : str, default="auto"
        Device to use for computation. If "auto", keeps data on its current device.
        Otherwise, temporarily moves data to specified device for computation.
        Output remains on the computation device.

    Returns
    -------
    C : pykeops.torch.LazyTensor
        If k is None, C is the full pairwise distance LazyTensor of shape (n_samples, m_samples).
        If k is provided, C is of shape (n_samples, k) containing the k smallest distances for each sample.
    indices : torch.Tensor or None
        If k is provided, indices is of shape (n_samples, k) containing the indices of the k nearest neighbors.
        Otherwise, None.
    """
    # Check metric support.
    if metric not in LIST_METRICS_KEOPS:
        raise ValueError(f"[TorchDR] ERROR : The '{metric}' distance is not supported.")

    # Move to computation device if needed
    # The moved tensors will be garbage collected after function returns
    if device != "auto" and str(X.device) != device:
        X = X.to(device)
        if Y is not None and Y is not X:
            Y = Y.to(device)

    # If Y is not provided, use X and decide about self–exclusion.
    if Y is None or Y is X:
        Y = X
        do_exclude_diag = exclude_diag
    else:
        do_exclude_diag = False  # Only exclude self when Y is not provided.

    # Create LazyTensors for pairwise operations.
    X_i = LazyTensor(X.unsqueeze(-2))  # Shape: (n, 1, d)
    Y_j = LazyTensor(Y.unsqueeze(-3))  # Shape: (1, m, d)

    # Compute pairwise distances.
    if metric == "sqeuclidean":
        C = ((X_i - Y_j) ** 2).sum(-1)
    elif metric == "euclidean":
        C = (((X_i - Y_j) ** 2).sum(-1)) ** 0.5
    elif metric == "manhattan":
        C = (X_i - Y_j).abs().sum(-1)
    elif metric == "angular":
        C = -(X_i | Y_j)
    else:
        raise ValueError(f"[TorchDR] ERROR : Unsupported metric '{metric}'.")

    # If requested, exclude self–neighbors by adding a large constant to the diagonal.
    if do_exclude_diag:
        n = X.shape[0]
        Id = identity_matrix(n, keops=True, device=X.device, dtype=X.dtype)
        C = C + Id * 1e12

    if k is not None:
        C_knn, indices = kmin(C, k=k, dim=1)
        # Output stays on computation device (not moved back)
        return C_knn, indices
    else:
        # Output stays on computation device (not moved back)
        return C, None
