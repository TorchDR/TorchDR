"""K-ary neighborhood preservation metric for dimensionality reduction evaluation."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import numpy as np
import torch
from typing import Union, Optional

from torchdr.utils import to_torch
from torchdr.distance import pairwise_distances, FaissConfig


def neighborhood_preservation(
    X: Union[torch.Tensor, np.ndarray],
    Z: Union[torch.Tensor, np.ndarray],
    K: int,
    metric: str = "euclidean",
    backend: Optional[Union[str, FaissConfig]] = None,
    device: Optional[str] = None,
):
    r"""Compute K-ary neighborhood preservation between input data and embeddings.

    This metric measures how well local neighborhood structure is preserved
    when reducing from high-dimensional input data (X) to low-dimensional embeddings (Z).

    Parameters
    ----------
    X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
        Original high-dimensional data.
    Z : torch.Tensor or np.ndarray of shape (n_samples, n_features_reduced)
        Reduced low-dimensional embeddings.
    K : int
        Neighborhood size (number of nearest neighbors to consider).
    metric : str, default='euclidean'
        Distance metric to use for computing nearest neighbors.
        Options: 'euclidean', 'sqeuclidean', 'manhattan', 'angular'.
    backend : {'keops', 'faiss', None} or FaissConfig, optional
        Backend to use for k-NN computation:
        - 'keops': Memory-efficient symbolic computations
        - 'faiss': Fast approximate nearest neighbors (recommended for large datasets)
        - None: Standard PyTorch operations
        - FaissConfig object: FAISS with custom configuration
    device : str, optional
        Device to use for computation. If None, uses input device.

    Returns
    -------
    score : float or torch.Tensor
        Average proportion of preserved neighbors across all points.
        Value between 0 and 1, where 1 indicates perfect preservation.
        Returns numpy float if inputs are numpy, torch.Tensor otherwise.

    Examples
    --------
    >>> import torch
    >>> from torchdr.eval.neighborhood_preservation import neighborhood_preservation
    >>>
    >>> # Generate example data
    >>> X = torch.randn(100, 50)  # High-dimensional data
    >>> Z = torch.randn(100, 2)   # Low-dimensional embedding
    >>>
    >>> # Compute preservation score
    >>> score = neighborhood_preservation(X, Z, K=10)
    >>> print(f"Neighborhood preservation: {score:.3f}")

    Notes
    -----
    The metric computes the Jaccard similarity (intersection over union) between
    the K-nearest neighbor sets in the original and reduced spaces for each point,
    then averages across all points.

    For large datasets, using backend='faiss' is recommended for efficiency.
    The metric excludes self-neighbors (i.e., the point itself).
    """
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}")

    input_is_numpy = not isinstance(X, torch.Tensor) or not isinstance(Z, torch.Tensor)

    X = to_torch(X)
    Z = to_torch(Z)

    if X.shape[0] != Z.shape[0]:
        raise ValueError(
            f"X and Z must have same number of samples, got {X.shape[0]} and {Z.shape[0]}"
        )

    n_samples = X.shape[0]

    if K >= n_samples:
        raise ValueError(f"K ({K}) must be less than number of samples ({n_samples})")

    if device is None:
        device = X.device
    else:
        device = torch.device(device)

    X = X.to(device)
    Z = Z.to(device)

    _, neighbors_X = pairwise_distances(
        X,
        metric=metric,
        backend=backend,
        k=K,
        exclude_diag=True,
        return_indices=True,
        device=device,
    )

    _, neighbors_Z = pairwise_distances(
        Z,
        metric=metric,
        backend=backend,
        k=K,
        exclude_diag=True,
        return_indices=True,
        device=device,
    )

    # Vectorized computation using broadcasting to check neighborhood overlap
    neighbors_X_expanded = neighbors_X.unsqueeze(2)  # (n_samples, K, 1)
    neighbors_Z_expanded = neighbors_Z.unsqueeze(1)  # (n_samples, 1, K)

    matches = (neighbors_X_expanded == neighbors_Z_expanded).any(
        dim=2
    )  # (n_samples, K)
    overlaps = matches.float().sum(dim=1) / K

    score = overlaps.mean()

    if input_is_numpy:
        score = score.detach().cpu().numpy().item()

    return score
