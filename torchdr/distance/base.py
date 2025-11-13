"""Distances based on various backends."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from typing import Optional, Union


from .torch import pairwise_distances_torch
from .keops import pairwise_distances_keops
from .faiss import pairwise_distances_faiss, FaissConfig


def pairwise_distances(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    metric: str = "euclidean",
    backend: Optional[Union[str, FaissConfig]] = None,
    exclude_diag: bool = False,
    k: Optional[int] = None,
    return_indices: bool = False,
    device: str = "auto",
):
    r"""Compute pairwise distances between two tensors.

    This is the main distance computation function that supports multiple backends
    for efficient computation. It can compute:
    - Full pairwise distance matrices between X and Y (or X and itself)
    - k-nearest neighbor distances when k is specified
    - Distances with various metrics (euclidean, manhattan, angular, etc.)

    For computing distances between specific indexed subsets, use
    pairwise_distances_indexed instead.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Input data.
    Y : torch.Tensor of shape (m_samples, n_features), optional
        Input data. If None, Y is set to X.
    metric : str, optional
        Metric to use. Default is "euclidean".
    backend : {'keops', 'faiss', None} or FaissConfig, optional
        Backend to use for computation. Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
        If None, use standard torch operations.
    exclude_diag : bool, optional
        Whether to exclude the diagonal from the distance matrix.
        Only used when k is not None. Default is False.
    k : int, optional
        If not None, return only the k-nearest neighbors.
    return_indices : bool, optional
        Whether to return the indices of the k-nearest neighbors.
        Default is False.
    device : str, default="auto"
        Device to use for computation. If "auto", keeps data on its current device.
        Otherwise, temporarily moves data to specified device for computation.
        Output remains on the computation device. Used with backend=None (torch) and backend="keops".

    Returns
    -------
    C : torch.Tensor
        Pairwise distances.
    indices : torch.Tensor, optional
        Indices of the k-nearest neighbors. Only returned if k is not None.

    Examples
    --------
    >>> import torch
    >>> from torchdr.distance import pairwise_distances, FaissConfig

    >>> # Basic usage
    >>> X = torch.randn(1000, 128)
    >>> distances = pairwise_distances(X, k=10, backend='faiss')

    >>> # Using FaissConfig with custom settings
    >>> config = FaissConfig(temp_memory=2.0)
    >>> distances = pairwise_distances(X.cuda(), k=10, backend=config)
    """
    # Parse backend parameter
    if isinstance(backend, FaissConfig):
        backend_str = "faiss"
        config = backend
    else:
        backend_str = backend
        config = None

    if backend_str == "keops":
        C, indices = pairwise_distances_keops(
            X=X, Y=Y, metric=metric, exclude_diag=exclude_diag, k=k, device=device
        )
    elif backend_str == "faiss":
        if k is not None:
            C, indices = pairwise_distances_faiss(
                X=X,
                Y=Y,
                metric=metric,
                k=k,
                exclude_diag=exclude_diag,
                config=config,
                device=device,
            )
        else:
            # Fall back to PyTorch when FAISS is specified but k is not provided
            C, indices = pairwise_distances_torch(
                X=X, Y=Y, metric=metric, k=k, exclude_diag=exclude_diag, device=device
            )
    else:
        C, indices = pairwise_distances_torch(
            X=X, Y=Y, metric=metric, k=k, exclude_diag=exclude_diag, device=device
        )

    if return_indices:
        return C, indices
    else:
        return C


def pairwise_distances_indexed(
    X: torch.Tensor,
    query_indices: Optional[torch.Tensor] = None,
    key_indices: Optional[torch.Tensor] = None,
    Y: Optional[torch.Tensor] = None,
    metric: str = "sqeuclidean",
    backend: Optional[Union[str, FaissConfig]] = None,
    device: str = "auto",
):
    r"""Compute pairwise distances between indexed subsets of tensors.

    This function efficiently computes distances between specific subsets of points
    selected by indices, rather than computing the full pairwise distance matrix.
    It's particularly useful for:
    - Computing distances to specific neighbors only (e.g., k-NN indices)
    - Multi-GPU scenarios where each GPU processes a chunk of data
    - Negative sampling where distances are needed only to sampled points

    The function allows flexible indexing of both query points (from X) and
    key points (from Y or X if Y is None).

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Input data containing query points.
    query_indices : torch.Tensor of shape (n_queries,) or (n_queries, k), optional
        Indices of rows from X to use as queries.
        - If 1D: selects rows X[query_indices] as queries
        - If 2D: for each row i, uses X[query_indices[i, :]] as multiple queries
        - If None: uses all rows of X as queries
    key_indices : torch.Tensor of shape (n_keys,) or (n_queries, n_keys), optional
        Indices of rows from Y (or X if Y is None) to use as keys.
        - If 1D: selects rows as keys for all queries
        - If 2D: for each query i, uses specific keys at key_indices[i, :]
        - If None: uses all rows of Y (or X) as keys
    Y : torch.Tensor of shape (m_samples, n_features), optional
        Input data containing key points. If None, uses X for keys.
    metric : str, optional
        Metric to use for distance computation. Default is "sqeuclidean".
        Supported: "sqeuclidean", "euclidean", "manhattan", "angular", "sqhyperbolic"
    backend : {'keops', 'faiss', None} or FaissConfig, optional
        Backend to use for computation. Currently only None (torch) is supported
        for indexed operations.
    device : str, default="auto"
        Device to use for computation.

    Returns
    -------
    distances : torch.Tensor
        Pairwise distances with shape determined by input indices:
        - query_indices=None, key_indices=None: (n_samples, m_samples)
        - query_indices=1D, key_indices=None: (n_queries, m_samples)
        - query_indices=None, key_indices=1D: (n_samples, n_keys)
        - query_indices=1D, key_indices=1D: (n_queries, n_keys)
        - query_indices=2D, key_indices=2D: (n_queries, n_keys_per_query)

    Examples
    --------
    >>> import torch
    >>> from torchdr.distance import pairwise_distances_indexed

    >>> # Compute distances from chunk to negatives (multi-GPU use case)
    >>> X = torch.randn(1000, 128)
    >>> chunk_indices = torch.arange(100, 200)  # GPU's chunk
    >>> neg_indices = torch.randint(0, 1000, (100, 5))  # Negative samples
    >>> distances = pairwise_distances_indexed(
    ...     X, query_indices=chunk_indices, key_indices=neg_indices
    ... )
    >>> distances.shape
    torch.Size([100, 5])
    """
    if Y is None:
        Y = X

    # Handle device placement
    if device != "auto":
        X = X.to(device)
        Y = Y.to(device)
        if query_indices is not None:
            query_indices = query_indices.to(device)
        if key_indices is not None:
            key_indices = key_indices.to(device)

    # Extract query points
    if query_indices is None:
        X_queries = X
    elif query_indices.dim() == 1:
        X_queries = X[query_indices]
    else:  # 2D indices
        raise NotImplementedError("2D query indices not yet supported")

    # Extract key points
    if key_indices is None:
        Y_keys = Y
    elif key_indices.dim() == 1:
        Y_keys = Y[key_indices.long()]
    elif key_indices.dim() == 2:
        # Each query has specific keys
        if query_indices is not None and query_indices.dim() == 1:
            # Ensure key_indices has same first dimension as number of queries
            assert key_indices.shape[0] == len(query_indices), (
                f"key_indices first dim {key_indices.shape[0]} must match number of queries {len(query_indices)}"
            )
        Y_keys = Y[
            key_indices.long()
        ]  # Shape: (n_queries, n_keys_per_query, n_features)
    else:
        raise ValueError(f"key_indices must be 1D or 2D, got {key_indices.dim()}D")

    # Compute distances based on shapes
    if Y_keys.dim() == 2:
        # Standard case: queries x keys
        if metric == "sqeuclidean":
            distances = torch.cdist(X_queries, Y_keys, p=2) ** 2
        elif metric == "euclidean":
            distances = torch.cdist(X_queries, Y_keys, p=2)
        elif metric == "manhattan":
            distances = torch.cdist(X_queries, Y_keys, p=1)
        elif metric == "angular":
            distances = -torch.mm(X_queries, Y_keys.t())
        elif metric == "sqhyperbolic":
            X_norm = (X_queries**2).sum(-1, keepdim=True)
            Y_norm = (Y_keys**2).sum(-1, keepdim=True).t()
            distances = torch.relu(torch.cdist(X_queries, Y_keys, p=2) ** 2)
            denom = (1 - X_norm) * (1 - Y_norm)
            distances = torch.arccosh(1 + 2 * (distances / denom) + 1e-8) ** 2
        else:
            raise NotImplementedError(
                f"Metric '{metric}' not implemented for indexed distances"
            )
    else:  # Y_keys.dim() == 3
        # Each query has specific keys
        if metric == "sqeuclidean":
            distances = torch.sum((X_queries.unsqueeze(1) - Y_keys) ** 2, dim=-1)
        elif metric == "euclidean":
            distances = torch.sum((X_queries.unsqueeze(1) - Y_keys) ** 2, dim=-1).sqrt()
        elif metric == "manhattan":
            distances = torch.sum(torch.abs(X_queries.unsqueeze(1) - Y_keys), dim=-1)
        elif metric == "angular":
            distances = -torch.sum(X_queries.unsqueeze(1) * Y_keys, dim=-1)
        elif metric == "sqhyperbolic":
            Y_keys_norm = (Y_keys**2).sum(-1)
            X_norm = (X_queries**2).sum(-1, keepdim=True)
            distances = torch.relu(
                torch.sum((X_queries.unsqueeze(1) - Y_keys) ** 2, dim=-1)
            )
            denom = (1 - X_norm) * (1 - Y_keys_norm)
            distances = torch.arccosh(1 + 2 * (distances / denom) + 1e-8) ** 2
        else:
            raise NotImplementedError(
                f"Metric '{metric}' not implemented for indexed distances"
            )

    return distances
