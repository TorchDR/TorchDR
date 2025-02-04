"""Ground metrics and distances."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from torchdr.utils.utils import identity_matrix, kmin

from .keops import LazyTensor, pykeops

LIST_METRICS_KEOPS = ["euclidean", "sqeuclidean", "manhattan", "angular", "hyperbolic"]


def pairwise_distances(
    X: torch.Tensor,
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    backend: str = None,
    k: int = None,
):
    r"""Compute pairwise distances matrix between points in two datasets.

    Returns the pairwise distance matrix as torch tensor or KeOps lazy tensor
    (if keops is True).

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        First dataset.
    Y : torch.Tensor of shape (m_samples, n_features), optional
        Second dataset. If None, Y = X.
    metric : str, optional
        Metric to use for computing distances. The default is "sqeuclidean".
    backend: {"keops", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    k : int, optional
        Number of nearest neighbors to consider for the distances.
        Default is None.

    Returns
    -------
    C : torch.Tensor or pykeops.torch.LazyTensor (if keops is True)
    of shape (n_samples, m_samples)
        Pairwise distances matrix.
    """
    if Y is None:
        Y = X

    if backend == "keops" and not pykeops:  # pykeops no installed
        raise ValueError(
            "[TorchDR] ERROR : pykeops is not installed. "
            "Please install it to use `backend=keops`."
        )

    if backend == "keops":
        C, indices = _pairwise_distances_keops(X, Y, metric, k=k)
    else:
        C, indices = _pairwise_distances_torch(X, Y, metric, k=k)

    return C, indices


def symmetric_pairwise_distances(
    X: torch.Tensor,
    metric: str,
    backend: str = None,
    exclude_self: float = None,
    k: int = None,
):
    r"""Compute pairwise distances matrix between points in a dataset.

    Return the pairwise distance matrix as torch tensor or KeOps lazy tensor
    (if keops is True). Supports batched input. The batch dimension should be the first.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features) or (n_batch, n_samples_batch, n_features)
        Input dataset.
    metric : str, optional
        Metric to use for computing distances. The default is "sqeuclidean".
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    exclude_self : float, optional
        If not None, adds weight on the diagonal of the distance matrix.
    k : int, optional
        Number of nearest neighbors to consider for the distances.

    Returns
    -------
    C : torch.Tensor or pykeops.torch.LazyTensor (if keops is True) of shape (n_samples, n_samples) or (n_batch, n_samples_batch, n_samples_batch)
        Pairwise distances matrix.
    indices: torch.Tensor of shape (n_samples, k) or (n_batch, n_samples_batch, k)
        Indices of the k nearest neighbors. If k is None, indices is None.
    """  # noqa E501
    if backend == "keops" and not pykeops:  # pykeops no installed
        raise ValueError(
            "[TorchDR] ERROR : pykeops is not installed. "
            "Please install it to use `backend=keops`."
        )

    if backend == "keops":
        C, indices = _pairwise_distances_keops(
            X, metric=metric, k=k, exclude_self=exclude_self
        )
    else:
        C, indices = _pairwise_distances_torch(
            X, metric=metric, k=k, exclude_self=exclude_self
        )

    return C, indices


def _pairwise_distances_torch(
    X: torch.Tensor,
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    k: int = None,
    exclude_self: bool = False,
):
    r"""Compute pairwise distances between points using PyTorch.

    When Y is not provided (i.e. computing distances within X) and
    `exclude_self` is True, the self–distance for each point (i.e. the diagonal)
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
    exclude_self : bool, default False
        If True and Y is not provided, the self–distance (diagonal elements) are set to infinity,
        excluding the self index from the k nearest neighbors.

    Returns
    -------
    C : torch.Tensor
        If k is None, C is the full pairwise distance matrix of shape (n_samples, m_samples).
        If k is provided, C is of shape (n_samples, k) containing the k smallest distances for each sample.
    indices : torch.Tensor or None
        If k is provided, indices is of shape (n_samples, k) containing the indices of the k nearest neighbors.
        Otherwise, None.
    """
    # Check metric support.
    if metric not in LIST_METRICS_KEOPS:
        raise ValueError(f"[TorchDR] ERROR : The '{metric}' distance is not supported.")

    # If Y is not provided, use X and decide about self–exclusion.
    if Y is None:
        Y = X
        do_exclude = exclude_self
    else:
        do_exclude = False  # Only exclude self when Y is not provided.

    # Compute pairwise distances.
    if metric == "sqeuclidean":
        X_norm = (X**2).sum(-1)
        Y_norm = (Y**2).sum(-1)
        C = X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * X @ Y.transpose(-1, -2)
    elif metric == "euclidean":
        X_norm = (X**2).sum(-1)
        Y_norm = (Y**2).sum(-1)
        C = X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * X @ Y.transpose(-1, -2)
        C = torch.clip(C, min=0.0).sqrt()  # Avoid negatives due to precision.
    elif metric == "manhattan":
        C = (X.unsqueeze(-2) - Y.unsqueeze(-3)).abs().sum(-1)
    elif metric == "angular":
        C = -X @ Y.transpose(-1, -2)
    elif metric == "hyperbolic":
        X_norm = (X**2).sum(-1)
        Y_norm = (Y**2).sum(-1)
        C = (
            X_norm.unsqueeze(-1) + Y_norm.unsqueeze(-2) - 2 * X @ Y.transpose(-1, -2)
        ) / (X[..., 0].unsqueeze(-1) * Y[..., 0].unsqueeze(-2))
    else:
        raise ValueError(f"[TorchDR] ERROR : Unsupported metric '{metric}'.")

    # If requested, exclude self–neighbors by setting the diagonal to infinity.
    if do_exclude:
        n = C.shape[0]
        diag_idx = torch.arange(n, device=C.device)
        C[diag_idx, diag_idx] = float("inf")

    if k is not None:
        C_knn, indices = kmin(C, k=k, dim=1)
        return C_knn, indices
    else:
        return C, None


def _pairwise_distances_keops(
    X: torch.Tensor,
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    k: int = None,
    exclude_self: bool = False,
):
    r"""Compute pairwise distances between points using KeOps LazyTensors.

    When Y is not provided (i.e. computing distances within X) and
    `exclude_self` is True, the self–distance for each point (diagonal)
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
    exclude_self : bool, default False
        If True and Y is not provided, the self–distance (diagonal entries) are set to infinity,
        excluding the self index from the k nearest neighbors.

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

    # If Y is not provided, use X and decide about self–exclusion.
    if Y is None:
        Y = X
        do_exclude = exclude_self
    else:
        do_exclude = False  # Only exclude self when Y is not provided.

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
    elif metric == "hyperbolic":
        C = ((X_i - Y_j) ** 2).sum(-1) / (X_i[0] * Y_j[0])
    else:
        raise ValueError(f"[TorchDR] ERROR : Unsupported metric '{metric}'.")

    # If requested, exclude self–neighbors by masking the diagonal.
    if do_exclude:
        n = X.shape[0]
        Id = identity_matrix(n, keops=True, device=X.device, dtype=X.dtype)
        C = C + Id * float("inf")

    if k is not None:
        C_knn, indices = kmin(C, k=k, dim=1)
        return C_knn, indices
    else:
        return C, None


def symmetric_pairwise_distances_indices(
    X: torch.Tensor,
    indices: torch.Tensor,
    metric: str = "sqeuclidean",
):
    r"""Compute pairwise distances for a subset of pairs given by indices.

    The output distance matrix has shape (n, k) and its (i,j) element is the
    distance between X[i] and Y[indices[i, j]].

    Parameters
    ----------
    X : torch.Tensor of shape (n, p)
        Input dataset.
    indices : torch.Tensor of shape (n, k)
        Indices of the pairs for which to compute the distances.
    metric : str, optional
        Metric to use for computing distances. The default is "sqeuclidean".

    Returns
    -------
    C_indices : torch.Tensor of shape (n, k)
        Pairwise distances matrix for the subset of pairs.
    indices : torch.Tensor of shape (n, k)
        Indices of the pairs for which to compute the distances.
    """
    X_indices = X[indices.int()]  # Shape (n, k, p)

    if metric == "sqeuclidean":
        C_indices = torch.sum((X.unsqueeze(1) - X_indices) ** 2, dim=-1)
    elif metric == "euclidean":
        C_indices = torch.sum((X.unsqueeze(1) - X_indices) ** 2, dim=-1).sqrt()
    elif metric == "manhattan":
        C_indices = torch.sum(torch.abs(X.unsqueeze(1) - X_indices), dim=-1)
    elif metric == "angular":
        C_indices = -torch.sum(X.unsqueeze(1) * X_indices, dim=-1)
    elif metric == "hyperbolic":
        C_indices = torch.sum((X.unsqueeze(1) - X_indices) ** 2, dim=-1) / (
            X[:, 0].unsqueeze(1) * X_indices[:, :, 0]
        )
    else:
        raise NotImplementedError(f"Metric '{metric}' is not (yet) implemented.")

    return C_indices, indices
