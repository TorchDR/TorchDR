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
    Y: torch.Tensor = None,
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
    X : torch.Tensor of shape (n_samples, n_features) or (n_batch, n_samples_batch, n_features)
        Input dataset.
    Y : torch.Tensor of shape (n_samples, n_features) or (n_batch, n_samples_batch, n_features), optional
        Second dataset. If None, computes the pairwise distances between X and itself.
    metric : str, optional
        Metric to use for computing distances. The default is "euclidean".
    keops : bool, optional
        If True, uses KeOps for computing the distances.
    add_diagonal : float, optional
        If not None, adds a mass on the diagonal of the distance matrix.

    Returns
    -------
    C : torch.Tensor or pykeops.torch.LazyTensor (if keops is True) of shape (n_samples, n_samples) or (n_batch, n_samples_batch, n_samples_batch)
        Pairwise distances matrix.
    """  # noqa E501
    assert metric in LIST_METRICS, f"The '{metric}' distance is not supported."

    if Y is None:
        Y = X

    if keops:  # recommended for large datasets
        X_i = LazyTensor(X.unsqueeze(-2))
        Y_j = LazyTensor(X.unsqueeze(-3))

        if metric == "euclidean":
            C = ((X_i - Y_j) ** 2).sum(-1)
        elif metric == "manhattan":
            C = (X_i - Y_j).abs().sum(-1)
        elif metric == "angular":
            C = -(X_i | Y_j)
        elif metric == "hyperbolic":
            C = ((X_i - Y_j) ** 2).sum(-1) / (X_i[0] * Y_j[0])

    else:
        if metric == "euclidean":
            X_norm = (X**2).sum(-1)
            Y_norm = (Y**2).sum(-1)
            C = (
                X_norm.unsqueeze(-1)
                + Y_norm.unsqueeze(-2)
                - 2 * X @ Y.transpose(-1, -2)
            )
        elif metric == "manhattan":
            C = (X.unsqueeze(-2) - Y.unsqueeze(-3)).abs().sum(-1)
        elif metric == "angular":
            C = -X @ Y.transpose(-1, -2)
        elif metric == "hyperbolic":
            X_norm = (X**2).sum(-1)
            Y_norm = (Y**2).sum(-1)
            C = (
                X_norm.unsqueeze(-1)
                + Y_norm.unsqueeze(-2)
                - 2 * X @ Y.transpose(-1, -2)
            ) / (X[..., 0].unsqueeze(-1) * Y[..., 0].unsqueeze(-2))

    if add_diagonal is not None:  # add mass on the diagonal
        I = identity_matrix(C.shape[-1], keops, X.device, X.dtype)
        C += add_diagonal * I

    return C
