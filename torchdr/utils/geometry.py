# -*- coding: utf-8 -*-
"""
Spaces and associated metrics
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from pykeops.torch import Vi, Vj


def pairwise_distances(X, metric="euclidean", keops=True):
    """
    Compute pairwise distances matrix between points in a dataset.
    Returns the pairwise distance matrix as LazyTensor.

    Parameters
    ----------
    X : tensor of shape (n_samples, n_features)
        Input dataset.
    metric : str, optional
        Metric to use for computing distances. The default is "euclidean".
    keops : bool, optional
        If True, uses KeOps for computing the distances. The default is True.

    Returns
    -------
    D : tensor or lazy tensor (if keops is True) of shape (n_samples, n_samples)
        Pairwise distances matrix.
    """
    if keops:
        X_i = Vi(X)  # (N, 1, D) LazyTensor, equivalent to LazyTensor(X[:,None,:])
        X_j = Vj(X)  # (1, N, D) LazyTensor, equivalent to LazyTensor(X[None,:,:])
    else:
        X_i = X[:, None, :]
        X_j = X[None, :, :]

    if metric == "euclidean":
        D = ((X_i - X_j) ** 2).sum(-1)
    elif metric == "manhattan":
        D = (X_i - X_j).abs().sum(-1)
    elif metric == "angular":
        D = -(X_i | X_j)
    elif metric == "hyperbolic":
        D = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])
    else:
        raise NotImplementedError(f"The '{metric}' distance is not supported.")

    return D
