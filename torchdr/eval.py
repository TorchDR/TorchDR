# -*- coding: utf-8 -*-
"""
Evaluation methods for dimensionality reduction
"""
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from torchdr.utils import to_torch, LIST_METRICS, pairwise_distances


def silhouette_samples(
    X: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    weights: torch.Tensor | np.ndarray = None,
    metric: str = "euclidean",
    device: str = None,
    keops: bool = True,
):
    """
    Compute the silhouette coefficients for each sample in :math:`\mathbf{X}`.
    Each coefficient is calculated using the mean intra-cluster
    distance (:math:`a`) and the mean nearest-cluster distance (:math:`b`) of
    the sample, according to the formula :math:`(b - a) / max(a, b)`.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
        Input data.
    labels : torch.Tensor or np.ndarray of shape (n_samples_x,)
        Labels associated to X.
    weights : torch.Tensor or np.ndarray of shape (n_samples_x,), optional
        Probability vector taking into account the relative importance
        of samples in X. The default is None and considers uniform weights.
    metric : str, optional
        The distance to use for computing pairwise distances. Must be an
        element of LIST_METRICS. The default is 'euclidean'.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.

    Returns
    -------
    coefficients : torch.Tensor or np.ndarray of shape (n_samples_x,)
        Silhouette coefficients for each sample.

    References
    ----------
    .. [23] Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the
            interpretation and validation of cluster analysis. Journal of
            computational and applied mathematics, 20, 53-65.

    """
    X = to_torch(X)
    labels = to_torch(labels)
    if weights is not None:
        weights = to_torch(weights)

    # compute intra and inter cluster distances by block
    unique_labels = torch.unique(labels)
    pos_labels = [torch.where(labels == label)[0] for label in unique_labels]
    A = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
    B = torch.full(X.shape[0], torch.inf, dtype=X.dtype, device=X.device)

    for i, pos_i in enumerate(pos_labels[:-1]):
        intra_cluster_dists = pairwise_distances(X[pos_i], X[pos_i], metric, keops)

        if weights is None:
            intra_cluster_dists = intra_cluster_dists.sum(axis=1) / (
                intra_cluster_dists.shape[0] - 1
            )
        else:
            matrix_weights_i = weights[pos_i].view(1, -1).repeat((pos_i.shape[0], 1))
            intra_cluster_dists = intra_cluster_dists * matrix_weights_i
            matrix_weights_i = matrix_weights_i.fill_diagonal_(0.0)
            intra_cluster_dists = intra_cluster_dists.sum(
                axis=1
            ) / matrix_weights_i.sum(axis=1)

        A[pos_i] = intra_cluster_dists

        for pos_j in pos_labels[i + 1 :]:
            inter_cluster_dists = pairwise_distances(X[pos_i], X[pos_j], metric, keops)

            if weights is None:
                dist_pos_i = inter_cluster_dists.mean(axis=1)
                dist_pos_j = inter_cluster_dists.mean(axis=0)
            else:
                dist_pos_i = inter_cluster_dists * weights[pos_j][None, :]
                dist_pos_j = inter_cluster_dists * weights[pos_i][:, None]
                dist_pos_i = dist_pos_i.sum(dim=1) / weights[pos_j].sum()
                dist_pos_j = dist_pos_j.sum(dim=0) / weights[pos_i].sum()

            B[pos_i] = torch.minimum(dist_pos_i, B[pos_i])
            B[pos_j] = torch.minimum(dist_pos_j, B[pos_j])

    # compute coefficients
    coefficients = (B - A) / torch.maximum(A, B)
    return torch.nan_to_num(coefficients, 0.0)


def silhouette_score(
    X: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    weights: torch.Tensor | np.ndarray = None,
    metric: str = "euclidean",
    device: str = None,
    keops: bool = True,
    sample_size: int = None,
    random_state: int = None,
):
    """

    Compute the Silhouette Score for all samples in :math:`\mathbf{X}` as the
    mean of silhouette coefficient of each sample.
    Each coefficient is calculated using the mean intra-cluster
    distance (:math:`a`) and the mean nearest-cluster distance (:math:`b`) of
    the sample, according to the formula :math:`(b - a) / max(a, b)`.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
        Input data.
    labels : torch.Tensor or np.ndarray of shape (n_samples_x,)
        Labels associated to X.
    weights : torch.Tensor or np.ndarray of shape (n_samples_x,), optional
        Probability vector taking into account the relative importance
        of samples in X. The default is None and considers uniform weights.
    metric : str, optional
        The distance to use for computing pairwise distances. Must be an
        element of LIST_METRICS. The default is 'euclidean'.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    sample_size : int, optional
        Number of samples to use when computing the score on a random subset.
        If sample_size is None, no sampling is used.
    random_state : int, optional
        Random state for selecting a subset of samples. Used when sample_size
        is not None.

    Returns
    -------
    silhouette_score : float
        mean silhouette coefficients for all samples.

    """  # noqa E501

    coefficients = silhouette_samples(X, labels, weights, metric, device, keops)
    silhouette_score = coefficients.mean()

    return silhouette_score
