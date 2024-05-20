# -*- coding: utf-8 -*-
"""
Ground metrics and distances
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from pykeops.torch import LazyTensor

from torchdr.utils.utils import identity_matrix

LIST_METRICS = ["euclidean", "manhattan", "angular", "hyperbolic"]


def pairwise_distances(
    X: torch.Tensor,
    metric: str = "euclidean",
    keops: bool = False,
    add_diagonal: float = 1e12,
):
    """
    Compute pairwise distances matrix between points in a dataset.
    Returns the pairwise distance matrix as tensor or lazy tensor (if keops is True).
    Supports batched input. The batch dimension should be the first.

    Parameters
    ----------
    X : tensor of shape (n_samples, n_features) or (batch_size, n_samples, n_features)
        Input dataset.
    metric : str, optional
        Metric to use for computing distances. The default is "euclidean".
    keops : bool, optional
        If True, uses KeOps for computing the distances.

    Returns
    -------
    D : tensor or lazy tensor (if keops is True) of shape (n_samples, n_samples)
        or (batch_size, n_samples, n_samples)
        Pairwise distances matrix.
    """
    assert metric in LIST_METRICS, f"The '{metric}' distance is not supported."

    if keops:  # recommended for large datasets
        X_i = LazyTensor(X.unsqueeze(-2))
        X_j = LazyTensor(X.unsqueeze(-3))

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
            D = (
                X_norm.unsqueeze(-1)
                + X_norm.unsqueeze(-2)
                - 2 * X @ X.transpose(-1, -2)
            )
        elif metric == "manhattan":
            D = (X.unsqueeze(-2) - X.unsqueeze(-3)).abs().sum(-1)
        elif metric == "angular":
            D = -X @ X.transpose(-1, -2)
        elif metric == "hyperbolic":
            X_norm = (X**2).sum(-1)
            D = (
                X_norm.unsqueeze(-1)
                + X_norm.unsqueeze(-2)
                - 2 * X @ X.transpose(-1, -2)
            ) / (X[..., 0].unsqueeze(-1) * X[..., 0].unsqueeze(-2))

    if add_diagonal is not None:  # add mass on the diagonal
        I = identity_matrix(D.shape[-1], keops, X.device, X.dtype)
        D += add_diagonal * I

    return D
