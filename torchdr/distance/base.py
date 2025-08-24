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
):
    r"""Compute pairwise distances between two tensors.

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

    >>> # With float16 precision for GPU
    >>> config = FaissConfig(use_float16=True)
    >>> distances = pairwise_distances(X.cuda(), k=10, backend=config)

    >>> # Using FaissConfig with custom settings
    >>> config = FaissConfig(use_float16=True, temp_memory=2.0)
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
            X=X, Y=Y, metric=metric, exclude_diag=exclude_diag, k=k
        )
    elif backend_str == "faiss":
        if k is not None:
            C, indices = pairwise_distances_faiss(
                X=X, Y=Y, metric=metric, k=k, exclude_diag=exclude_diag, config=config
            )
        else:
            raise ValueError(
                "[TorchDR] ERROR : k must be provided when using `backend=faiss`."
            )
    else:
        C, indices = pairwise_distances_torch(
            X=X, Y=Y, metric=metric, k=k, exclude_diag=exclude_diag
        )

    if return_indices:
        return C, indices
    else:
        return C


def symmetric_pairwise_distances_indices(
    X: torch.Tensor,
    indices: torch.Tensor,
    metric: str = "sqeuclidean",
    return_indices: bool = False,
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
    return_indices : bool, optional
        Whether to return the indices of the pairs for which to compute the distances.
        Default is False.

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
    elif metric == "sqhyperbolic":
        X_indices_norm = (X_indices**2).sum(-1)
        X_norm = (X**2).sum(-1)
        C_indices = torch.relu(torch.sum((X.unsqueeze(1) - X_indices) ** 2, dim=-1))
        denom = (1 - X_norm).unsqueeze(-1) * (1 - X_indices_norm)
        C_indices = torch.arccosh(1 + 2 * (C_indices / denom) + 1e-8) ** 2
    else:
        raise NotImplementedError(f"Metric '{metric}' is not (yet) implemented.")

    if return_indices:
        return C_indices, indices
    else:
        return C_indices
