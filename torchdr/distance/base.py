"""Distances based on various backends."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from typing import Optional

from torchdr.utils.wrappers import compile_if_requested


from .torch import pairwise_distances_torch
from .keops import pairwise_distances_keops
from .faiss import pairwise_distances_faiss


@compile_if_requested
def pairwise_distances(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    metric: str = "euclidean",
    backend: Optional[str] = None,
    exclude_diag: bool = False,
    k: Optional[int] = None,
    compile: bool = False,
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
    backend : {'keops', 'faiss', None}, optional
        Backend to use for computation.
        If None, use standard torch operations.
    exclude_diag : bool, optional
        Whether to exclude the diagonal from the distance matrix.
        Only used when k is not None. Default is False.
    k : int, optional
        If not None, return only the k-nearest neighbors.
    compile : bool, default=False
        Whether to use torch.compile for faster computation.

    Returns
    -------
    C : torch.Tensor
        Pairwise distances.
    indices : torch.Tensor, optional
        Indices of the k-nearest neighbors. Only returned if k is not None.
    """
    if backend == "keops":
        C, indices = pairwise_distances_keops(
            X=X, Y=Y, metric=metric, exclude_diag=exclude_diag, k=k
        )
    elif backend == "faiss":
        if k is not None:
            C, indices = pairwise_distances_faiss(
                X=X, Y=Y, metric=metric, k=k, exclude_diag=exclude_diag
            )
        else:
            raise ValueError(
                "[TorchDR] ERROR : k must be provided when using `backend=faiss`."
            )
    else:
        C, indices = pairwise_distances_torch(
            X=X, Y=Y, metric=metric, k=k, exclude_diag=exclude_diag, compile=compile
        )

    return C, indices


@compile_if_requested
def symmetric_pairwise_distances_indices(
    X: torch.Tensor,
    indices: torch.Tensor,
    metric: str = "sqeuclidean",
    compile: bool = False,
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
    compile : bool, optional
        Whether to compile the distance computation. Default is False.

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

    return C_indices, indices
