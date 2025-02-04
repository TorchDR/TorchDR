"""Ground metrics and distances."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np

from torchdr.utils.utils import identity_matrix, kmin

from .keops import LazyTensor, pykeops
from .faiss import faiss

LIST_METRICS_KEOPS = ["euclidean", "sqeuclidean", "manhattan", "angular", "hyperbolic"]


def pairwise_distances(
    X: torch.Tensor,
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    backend: str = None,
    exclude_self: bool = False,
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
    exclude_self : bool, optional
        If True, adds weight on the diagonal of the distance matrix.
        Default is False.
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

    elif backend == "faiss" and not faiss:  # faiss not installed
        raise ValueError(
            "[TorchDR] ERROR : faiss is not installed. "
            "Please install it to use `backend=faiss`."
        )

    if backend == "keops":
        C, indices = _pairwise_distances_keops(
            X, Y, metric, k=k, exclude_self=exclude_self
        )
    elif backend == "faiss":
        C, indices = _pairwise_distances_faiss(
            X, Y, metric, k=k, exclude_self=exclude_self
        )
    else:
        C, indices = _pairwise_distances_torch(
            X, Y, metric, k=k, exclude_self=exclude_self
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
        C[diag_idx, diag_idx] = 1e12

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
        C = C + Id * 1e12

    if k is not None:
        C_knn, indices = kmin(C, k=k, dim=1)
        return C_knn, indices
    else:
        return C, None


def _pairwise_distances_faiss(
    X: torch.Tensor,
    Y: torch.Tensor = None,
    metric: str = "euclidean",
    k: int = None,
    exclude_self: bool = False,
):
    r"""Compute the k nearest neighbors using FAISS.

    Supported metrics are:
      - "euclidean": returns the Euclidean distance (square root of the squared distance)
      - "sqeuclidean": returns the squared Euclidean distance (as computed by FAISS)
      - "angular": returns the negative inner-product (after normalizing vectors)

    If Y is not provided then we assume a self–search and, if `exclude_self` is True,
    the self–neighbor is removed from the results.

    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
        Query dataset.
    Y : torch.Tensor of shape (m, d), optional
        Database dataset. If None, Y is set equal to X.
    metric : str, default "euclidean"
        One of "euclidean", "sqeuclidean", or "angular".
    k : int, optional
        Number of nearest neighbors to return.
        (If `exclude_self` is True in a self–search, then k+1 neighbors are retrieved first.)
    exclude_self : bool, default False
        When True and Y is not provided (i.e. self–search), the self–neighbor (index i for query i)
        is excluded from the k results.

    Returns
    -------
    distances : torch.Tensor of shape (n, k)
        Nearest neighbor distances.
        For metric=="euclidean", distances are Euclidean (i.e. square root of L2^2).
        For metric=="sqeuclidean", distances are the squared Euclidean distances.
        For metric=="angular", distances are the (normalized) inner product scores.
    indices : torch.Tensor of shape (n, k)
        Indices of the k nearest neighbors.
    """
    if metric not in {"euclidean", "sqeuclidean", "angular"}:
        raise ValueError(
            "Only 'euclidean', 'sqeuclidean', and 'angular' metrics are supported for FAISS."
        )

    # If Y is not provided, we are searching within X.
    if Y is None:
        Y = X
        self_search = True
    else:
        self_search = False

    # Convert input tensors to numpy arrays of type float32.
    X_np = X.detach().cpu().numpy().astype(np.float32)
    Y_np = Y.detach().cpu().numpy().astype(np.float32)
    n, d = X_np.shape

    # Set up the FAISS index depending on the metric.
    if metric == "angular":
        # Normalize vectors for angular similarity.
        X_norm = np.linalg.norm(X_np, axis=1, keepdims=True)
        X_norm[X_norm == 0] = 1.0
        X_np = X_np / X_norm

        Y_norm = np.linalg.norm(Y_np, axis=1, keepdims=True)
        Y_norm[Y_norm == 0] = 1.0
        Y_np = Y_np / Y_norm

        index = faiss.IndexFlatIP(d)
    elif metric in {"euclidean", "sqeuclidean"}:
        # Use the L2 index. Note: FAISS returns squared distances.
        index = faiss.IndexFlatL2(d)
    else:
        # This branch should never be reached due to the initial check.
        raise ValueError(f"Metric '{metric}' is not supported.")

    # If the input tensor is on GPU, move the index to GPU.
    if X.device.type == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Add the database vectors to the index.
    index.add(Y_np)

    # If self-search and excluding self, search for one extra neighbor.
    if self_search and exclude_self:
        k_search = k + 1
    else:
        k_search = k

    # Perform the search.
    D, Ind = index.search(X_np, k_search)  # D: (n, k_search), I: (n, k_search)

    # For "euclidean", take the square root of the squared distances.
    if metric == "euclidean":
        D = np.sqrt(D)
    # For "sqeuclidean", leave the distances as returned (i.e. squared).

    # If doing self–search with self–exclusion, remove the self neighbor from the results.
    if self_search and exclude_self:
        new_D = []
        new_Ind = []
        for i in range(n):
            row_indices = Ind[i]
            row_distances = D[i]
            # Exclude the query itself.
            mask = row_indices != i
            filtered_indices = row_indices[mask]
            filtered_distances = row_distances[mask]
            new_Ind.append(filtered_indices[:k])
            new_D.append(filtered_distances[:k])
        D = np.vstack(new_D)
        Ind = np.vstack(new_Ind)
    else:
        # Otherwise, if extra neighbors were retrieved, trim to k.
        if k_search > k:
            D = D[:, :k]
            Ind = Ind[:, :k]

    # Convert back to torch tensors.
    device = X.device
    distances = torch.from_numpy(D).to(device)
    indices = torch.from_numpy(Ind).to(device).long()

    return distances, indices


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
