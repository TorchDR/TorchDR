# -*- coding: utf-8 -*-
"""Evaluation methods for dimensionality reduction."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
import warnings
from random import sample, seed

from torchdr.utils import to_torch, pairwise_distances, prod_matrix_vector


admissible_LIST_METRICS = ["euclidean", "manhattan", "hyperbolic", "precomputed"]


def silhouette_samples(
    X: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    weights: torch.Tensor | np.ndarray = None,
    metric: str = "euclidean",
    device: str = None,
    keops: bool = True,
    warn: bool = True,
):
    r"""Compute the silhouette coefficients for each data sample.

    Each coefficient is calculated using the mean intra-cluster
    distance (:math:`a`) and the mean nearest-cluster distance (:math:`b`) of
    the sample, according to the formula :math:`(b - a) / max(a, b)`.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Parameters
    ----------
    X : torch.Tensor, np.ndarray of shape (n_samples_x, n_samples_x) if
        `metric="precomputed" else (n_samples_x, n_features)
        Input data as a pairwise distance matrix or a feature matrix.
    labels : torch.Tensor or np.ndarray of shape (n_samples_x,)
        Labels associated to X.
    weights : torch.Tensor or np.ndarray of shape (n_samples_x,), optional
        Probability vector taking into account the relative importance
        of samples in X. The default is None and considers uniform weights.
    metric : str, optional
        The distance to use for computing pairwise distances. Must be an
        element of ["euclidean", "manhattan", "hyperbolic", "precomputed"].
        The default is 'euclidean'.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    warn : bool, optional
        Whether to output warnings when edge cases are identified.

    Returns
    -------
    coefficients : torch.Tensor or np.ndarray of shape (n_samples_x,)
        Silhouette coefficients for each sample.

    References
    ----------
    .. [24] Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the
            interpretation and validation of cluster analysis. Journal of
            computational and applied mathematics, 20, 53-65.

    """
    if metric not in admissible_LIST_METRICS:
        raise ValueError(f"metric = {metric} must be in {admissible_LIST_METRICS}")

    if metric == "precomputed":
        if X.shape[0] != X.shape[1]:
            raise ValueError("X must be a square matrix with metric = 'precomputed'")
        if keops and warn:
            warnings.warn(
                "[TorchDR] WARNING : keops not supported with metric = 'precomputed'",
                stacklevel=2,
            )

    X = to_torch(X)
    labels = to_torch(labels)
    if weights is not None:
        weights = to_torch(weights)

    if device is None:
        device = X.device

    # compute intra and inter cluster distances by block
    unique_labels = torch.unique(labels)
    pos_labels = [torch.where(labels == label)[0] for label in unique_labels]
    A = torch.zeros((X.shape[0],), dtype=X.dtype, device=device)
    B = torch.full((X.shape[0],), torch.inf, dtype=X.dtype, device=device)

    for i, pos_i in enumerate(pos_labels):
        if pos_i.shape[0] > 1:
            if metric == "precomputed":
                intra_cluster_dists = X[pos_i, :][:, pos_i]
            else:
                intra_cluster_dists = pairwise_distances(
                    X[pos_i], X[pos_i], metric, keops
                )

            if weights is None:
                intra_cluster_dists = intra_cluster_dists.sum(1).squeeze(-1) / (
                    pos_i.shape[0] - 1
                )
            else:
                intra_cluster_dists = (
                    prod_matrix_vector(intra_cluster_dists, weights[pos_i])
                    .sum(dim=1)
                    .squeeze(-1)
                )
                sub_weights_i = (
                    torch.full(
                        (pos_i.shape[0],),
                        weights[pos_i].sum(),
                        dtype=X.dtype,
                        device=device,
                    )
                    - weights[pos_i]
                )
                intra_cluster_dists = intra_cluster_dists / sub_weights_i

        else:
            intra_cluster_dists = 0.0
            if warn:
                warnings.warn(
                    "[TorchDR] WARNING : ill-defined intra-cluster mean distance "
                    "as one cluster contains only one sample.",
                    stacklevel=2,
                )
        A[pos_i] = intra_cluster_dists

        for pos_j in pos_labels[i + 1 :]:
            if metric == "precomputed":
                inter_cluster_dists = X[pos_i, :][:, pos_j]
            else:
                inter_cluster_dists = pairwise_distances(
                    X[pos_i], X[pos_j], metric, keops
                )

            if weights is None:
                dist_pos_i = inter_cluster_dists.sum(1).squeeze(-1) / pos_j.shape[0]
                dist_pos_j = inter_cluster_dists.sum(0).squeeze(-1) / pos_i.shape[0]
            else:
                dist_pos_i = (
                    prod_matrix_vector(inter_cluster_dists, weights[pos_j], True)
                    .sum(1)
                    .squeeze(-1)
                )
                dist_pos_j = (
                    prod_matrix_vector(inter_cluster_dists, weights[pos_i])
                    .sum(0)
                    .squeeze(-1)
                )
                dist_pos_i = dist_pos_i / weights[pos_j].sum()
                dist_pos_j = dist_pos_j / weights[pos_i].sum()

            B[pos_i] = torch.minimum(dist_pos_i, B[pos_i])
            B[pos_j] = torch.minimum(dist_pos_j, B[pos_j])

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
    warn: bool = True,
):
    r"""Compute the Silhouette score as the mean of silhouette coefficients.

    Each coefficient is calculated using the mean intra-cluster
    distance (:math:`a`) and the mean nearest-cluster distance (:math:`b`) of
    the sample, according to the formula :math:`(b - a) / max(a, b)`.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Parameters
    ----------
    X : torch.Tensor, np.ndarray of shape (n_samples_x, n_samples_x) if
        `metric="precomputed" else (n_samples_x, n_features)
        Input data as a pairwise distance matrix or a feature matrix.
    labels : torch.Tensor or np.ndarray of shape (n_samples_x,)
        Labels associated to X.
    weights : torch.Tensor or np.ndarray of shape (n_samples_x,), optional
        Probability vector taking into account the relative importance
        of samples in X. The default is None and considers uniform weights.
    metric : str, optional
        The distance to use for computing pairwise distances. Must be an
        element of ["euclidean", "manhattan", "hyperbolic", "precomputed"].
        The default is 'euclidean'.
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
    warn : bool, optional
        Whether to output warnings when edge cases are identified.

    Returns
    -------
    silhouette_score : float
        mean silhouette coefficients for all samples.

    References
    ----------
    .. [24] Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the
            interpretation and validation of cluster analysis. Journal of
            computational and applied mathematics, 20, 53-65.
    """
    if sample_size is None:
        coefficients = silhouette_samples(
            X, labels, weights, metric, device, keops, warn
        )
    else:
        seed(random_state)
        indices = sample(range(X.shape[0]), sample_size)
        sub_weights = None if weights is None else weights[indices]

        if metric == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    "X must be a square matrix with metric = 'precomputed'"
                )

            sub_X = X[indices, :][:, indices]
        else:
            sub_X = X[indices]

        coefficients = silhouette_samples(
            sub_X, labels[indices], sub_weights, metric, device, keops, warn
        )

    silhouette_score = coefficients.mean()

    return silhouette_score
