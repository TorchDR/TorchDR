# -*- coding: utf-8 -*-
"""
Spaces and associated metrics
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from pykeops.torch import Vi, Vj

from torchdr.utils.utils import identity_matrix

LIST_METRICS = ["euclidean", "manhattan", "angular", "hyperbolic"]


# inspired from the benchmark_KNN of KeOps
def pairwise_distances(
    X: torch.Tensor,
    metric: str = "euclidean",
    keops: bool = False,
    add_diagonal: float = 1e12,
):
    """
    Compute pairwise distances matrix between points in a dataset.
    Returns the pairwise distance matrix as tensor or lazy tensor (if keops is True).

    Parameters
    ----------
    X : tensor of shape (n_samples, n_features)
        Input dataset.
    metric : str, optional
        Metric to use for computing distances. The default is "euclidean".
    keops : bool, optional
        If True, uses KeOps for computing the distances.

    Returns
    -------
    D : tensor or lazy tensor (if keops is True) of shape (n_samples, n_samples)
        Pairwise distances matrix.
    """
    assert metric in LIST_METRICS, f"The '{metric}' distance is not supported."

    if keops:  # recommended for large datasets
        X_i = Vi(X)  # (N, 1, D) LazyTensor, equivalent to LazyTensor(X[:,None,:])
        X_j = Vj(X)  # (1, N, D) LazyTensor, equivalent to LazyTensor(X[None,:,:])

        if metric == "euclidean":
            D = ((X_i - X_j) ** 2).sum(-1)
        elif metric == "manhattan":
            D = (X_i - X_j).abs().sum(-1)
        elif metric == "angular":
            D = -(X_i | X_j)
        elif metric == "hyperbolic":
            D = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])

    else:
        if metric == "euclidean":
            X_norm = (X**2).sum(-1)
            D = X_norm[:, None] + X_norm[None, :] - 2 * X @ X.T
        elif metric == "manhattan":
            D = (X[:, None, :] - X[None, :, :]).abs().sum(-1)
        elif metric == "angular":
            D = -X @ X.T
        elif metric == "hyperbolic":
            X_norm = (X**2).sum(-1)
            D = (X_norm[:, None] + X_norm[None, :] - 2 * X @ X.T) / (
                X[:, 0][:, None] * X[:, 0][None, :]
            )

    if add_diagonal is not None:  # add mass on the diagonal
        D += add_diagonal * identity_matrix(X.shape[0], keops, X.device, X.dtype)

    return D
