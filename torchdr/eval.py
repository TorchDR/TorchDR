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
import warnings
from random import sample, seed

from torchdr.utils import (
    to_torch,
    pairwise_distances,
    prod_matrix_vector,
    identity_matrix,
    kmin,
)


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
    r"""
    Compute the silhouette coefficients for each sample in :math:`\mathbf{X}`.
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
    r"""
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


def trustworthiness(
    X: torch.Tensor | np.ndarray,
    Z: torch.Tensor | np.ndarray,
    n_neighbors: int = 5,
    metric: str = "euclidean",
    device: str = None,
):
    r"""Compute the Trustworthiness within [0, 1] indicating to which extent
    the local structure is maintained in the embedding.

    The Trustworthiness is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    .. note:: This function cannot benefit from Keops.

    Parameters
    ----------
    X : torch.Tensor, np.ndarray of shape (n_samples_x, n_samples_x) if
        `metric="precomputed" else (n_samples_x, n_features_x)
        Data in the input space as a pairwise distance matrix
        or a feature matrix.
    Z : torch.Tensor, np.ndarray of shape (n_samples_x, n_samples_x) if
        `metric="precomputed" else (n_samples_x, n_features_z)
        Data in the embedding space as a pairwise distance matrix
        or a feature matrix.
    n_neighbors : int, default=5
        The number of neighbors that will be considered. Must satisfy
        `n_neighbors < n_samples / 2` to ensure the score to lie in [0, 1].
    metric : str, optional
        The distance to use for computing pairwise distances. Must be an
        element of ["euclidean", "manhattan", "hyperbolic", "precomputed"].
        The default is 'euclidean'.
    device : str, optional
        Device to use for computations.

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.

    References
    ----------
    .. [25] J.Venna and S.Kaski. Neighborhood Preservation in Nonlinear
        Projection Methods: An Experimental Study. In Proceedings of the
        International Conference on Artificial Neural Networks (ICANN 2001).

    """
    n_samples_x = X.shape[0]

    if n_neighbors >= n_samples_x / 2:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be less than n_samples_x / 2"
            f" ({n_samples_x / 2})"
        )

    if metric == "precomputed":
        if X.shape != (n_samples_x, n_samples_x):
            raise ValueError("X must be a square matrix with metric = 'precomputed'")
        if Z.shape != (n_samples_x, n_samples_x):
            raise ValueError("Z must be a square matrix with metric = 'precomputed'")

    if device is None:
        device = X.device

    if metric == "precomputed":
        CX = X
        CZ = Z
    else:
        CX = pairwise_distances(X, X, metric, keops=False)
        CZ = pairwise_distances(Z, Z, metric, keops=False)

    # we set the diagonal to high values to exclude the points themselves from
    # their own neighborhood
    I = identity_matrix(CX.shape[-1], False, X.device, X.dtype)
    CX = CX + (2.0 * CX.max()) * I
    CZ = CZ + (2.0 * CZ.max()) * I

    # sort values in the input space
    # need to find a way to avoid storing the full matrix as follows
    sorted_indices_X = torch.argsort(CX, dim=1)
    # get indices of nearest neighbors in the embedding space
    _, minK_indices_Z = kmin(CZ, k=n_neighbors, dim=1)

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    inverted_indices_X = torch.zeros((n_samples_x, n_samples_x), dtype=torch.int32)
    ordered_indices_X = torch.arange(n_samples_x)
    inverted_indices_X[ordered_indices_X[:, None], sorted_indices_X] = (
        ordered_indices_X + 1
    )
    ranks = inverted_indices_X[ordered_indices_X[:, None], minK_indices_Z] - n_neighbors
    score = torch.sum(ranks[ranks > 0])
    score = 1.0 - score * (
        2.0
        / (n_samples_x * n_neighbors * (2.0 * n_samples_x - 3.0 * n_neighbors - 1.0))
    )
    return score


def Kary_preservation_score(
    X: torch.Tensor | np.ndarray,
    Z: torch.Tensor | np.ndarray,
    K: int = 5,
    adjusted: bool = False,
    metric: str = "euclidean",
    device: str = None,
    keops: bool = True,
):
    r"""Compute the average agreement rate within [0, 1] between K-ary
    neighbourhoods in the input space `X` and the output space `Z`.

    The K-ary neighbourhoods in one space e.g `X` coincides with the set defined,
    for each sample i, as :math:`n^{X, K}_i = \{ j| 1 \leq r^{X}_{i, j} \leq K \}`
    where for any j, :math:`r^{X}_{i, j}` is the number of samples in a ball
    centered in `x_i` of radius `C^X(x_i, x_j)`. The average agreement between
    both spaces then reads as

    .. math::

        S = \frac{1}{NK} \sum^N_{i=1} |n^{X, K}_i \cap n^{Z, K}_i|

    It can be further adjusted for chance considering the following score:

    .. math::

        S_{adjusted} = \frac{(N-1)S - K}{N - 1 - K}

    Parameters
    ----------
    X : torch.Tensor, np.ndarray of shape (n_samples_x, n_samples_x) if
        `metric="precomputed" else (n_samples_x, n_features_x)
        Data in the input space as a pairwise distance matrix
        or a feature matrix.
    Z : torch.Tensor, np.ndarray of shape (n_samples_x, n_samples_x) if
        `metric="precomputed" else (n_samples_x, n_features_z)
        Data in the embedding space as a pairwise distance matrix
        or a feature matrix.
    K : int, default=5
        Maximum number of neighbors that will be considered. Must satisfy
        `n_neighbors < n_samples`.
    adjusted : bool, default is False.
        Whether to adjust the score for chance.
    metric : str, optional
        The distance to use for computing pairwise distances. Must be an
        element of ["euclidean", "manhattan", "hyperbolic", "precomputed"].
        The default is 'euclidean'.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.

    Returns
    -------
    score : float
        Average agreement rate between K-ary neighborhoods.

    References
    ----------
    .. [26] J.Lee, M.Verleysen, Quality assessment of dimensionality reduction:
        rank-based criteria, Neurocomputing (2009).

    """
    n_samples_x = X.shape[0]

    if K < n_samples_x:
        raise ValueError(f"K ({K}) must be less than n_samples_x ({n_samples_x})")

    if metric == "precomputed":
        if X.shape != (n_samples_x, n_samples_x):
            raise ValueError("X must be a square matrix with metric = 'precomputed'")
        if Z.shape != (n_samples_x, n_samples_x):
            raise ValueError("Z must be a square matrix with metric = 'precomputed'")

    if device is None:
        device = X.device

    if metric == "precomputed":
        CX = X
        CZ = Z
    else:
        CX = pairwise_distances(X, X, metric, keops)
        CZ = pairwise_distances(Z, Z, metric, keops)

    # we set the diagonal to high values to exclude the points themselves from
    # their own neighborhood
    I = identity_matrix(CX.shape[-1], False, X.device, X.dtype)
    CX = CX + (2.0 * CX.max()) * I
    CZ = CZ + (2.0 * CZ.max()) * I

    # get indices of nearest neighbors in the input space
    minK_values_X, minK_indices_X = kmin(CX, k=K, dim=1)
    # get indices of nearest neighbors in the embedding space
    minK_values_Z, minK_indices_Z = kmin(CZ, k=K, dim=1)

    # to handle equality cases w.r.t distances we have to iteratively reduce
    # radii of balls if the number of neighbors exceed K
    radii_idx_X = (K - 1) * torch.ones(CX.shape[0], device=device, dtype=torch.int32)
    ranks_above_K = True
    while ranks_above_K:
        neighborhoods_X = CX <= minK_values_X[:, radii_idx_X]
        sizes_X = neighborhoods_X.sum(dim=1)
        too_large_neighborhoods = torch.where(sizes_X > K)[0]
        if too_large_neighborhoods.shape[0] == 0:
            ranks_above_K = False
        else:
            radii_idx_X[too_large_neighborhoods] -= 1

    radii_idx_Z = (K - 1) * torch.ones(CZ.shape[0], device=device, dtype=torch.int32)
    ranks_above_K = True
    while ranks_above_K:
        neighborhoods_Z = CZ <= minK_values_Z[:, radii_idx_Z]
        sizes_Z = neighborhoods_Z.sum(dim=1)
        too_large_neighborhoods = torch.where(sizes_Z > K)[0]
        if too_large_neighborhoods.shape[0] == 0:
            ranks_above_K = False
        else:
            radii_idx_Z[too_large_neighborhoods] -= 1

    # compute the intersection between sets
    intersection_counts = [
        torch.isin(neighborhoods_X[i], neighborhoods_Z[i]).sum().item()
        for i in range(n_samples_x)
    ]

    score = sum(intersection_counts) / (K * n_samples_x)

    if adjusted:
        score = ((n_samples_x - 1.0) * score - K) / (n_samples_x - 1 - K)

    return score
