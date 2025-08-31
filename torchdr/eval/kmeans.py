"""K-means clustering evaluation for dimensionality reduction."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import warnings
import numpy as np
import torch
from typing import Union, Optional

from torchdr.utils import to_torch
from torchdr.utils.faiss import faiss

try:
    from torchmetrics.clustering import AdjustedRandScore
except ImportError:
    AdjustedRandScore = None


def kmeans_ari(
    X: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    n_clusters: Optional[int] = None,
    niter: int = 20,
    nredo: int = 1,
    device: Optional[str] = None,
    random_state: Optional[int] = None,
    verbose: bool = False,
):
    r"""Perform K-means clustering and compute Adjusted Rand Index.

    This function clusters the input data using FAISS K-means and computes
    the Adjusted Rand Index (ARI) between the predicted clusters and true labels.
    The ARI measures the similarity between two clusterings, adjusted for chance.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
        Input data to cluster.
    labels : torch.Tensor or np.ndarray of shape (n_samples,)
        True labels for computing ARI.
    n_clusters : int, optional
        Number of clusters. If None, uses the number of unique labels.
    niter : int, default=20
        Maximum number of K-means iterations.
    nredo : int, default=1
        Number of times to run K-means with different initializations,
        keeping the best result (lowest objective).
    device : str, optional
        Device to use for ARI computation. If None, uses the input device.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress information.

    Returns
    -------
    ari_score : float or torch.Tensor
        Adjusted Rand Index between predicted clusters and true labels.
        Values range from -1 to 1, where 1 indicates perfect agreement,
        0 indicates random labeling, and negative values indicate
        systematic disagreement. Returns numpy float if inputs are numpy,
        torch.Tensor if inputs are torch.
    predicted_labels : np.ndarray or torch.Tensor of shape (n_samples,)
        Cluster assignments from K-means. Returns same type as input X.

    Raises
    ------
    ImportError
        If FAISS or torchmetrics is not installed.
    ValueError
        If n_clusters is less than 1 or greater than n_samples.

    Examples
    --------
    >>> import torch
    >>> from torchdr.eval.kmeans import kmeans_ari
    >>>
    >>> # Generate sample data
    >>> X = torch.randn(1000, 50)
    >>> true_labels = torch.randint(0, 5, (1000,))
    >>>
    >>> # Compute ARI score
    >>> ari_score, pred_labels = kmeans_ari(X, true_labels)
    >>> print(f"ARI Score: {ari_score:.3f}")

    Notes
    -----
    The Adjusted Rand Index is a measure of clustering quality that:
    - Accounts for chance agreement between clusterings
    - Is symmetric (swapping predicted and true labels gives same result)
    - Has expected value of 0 for random clusterings
    - Has maximum value of 1 for identical clusterings

    FAISS K-means uses Lloyd's algorithm with optional multiple runs.
    GPU acceleration is automatically used if FAISS-GPU is installed and X is on GPU.
    """
    if faiss is False:
        raise ImportError(
            "[TorchDR] FAISS is required for kmeans_ari but not installed. "
            "Install it with: conda install -c pytorch -c nvidia faiss-gpu"
        )

    if AdjustedRandScore is None:
        raise ImportError(
            "[TorchDR] torchmetrics is required for kmeans_ari but not installed. "
            "Install it with: pip install torchmetrics"
        )

    input_is_numpy = not isinstance(X, torch.Tensor) or not isinstance(
        labels, torch.Tensor
    )

    X = to_torch(X)
    labels = to_torch(labels).squeeze()

    if device is None:
        device = X.device
    else:
        device = torch.device(device)

    X_np = X.detach().cpu().numpy().astype(np.float32)
    labels_np = labels.detach().cpu().numpy()

    n_samples, d = X_np.shape

    if n_clusters is None:
        n_clusters = len(np.unique(labels_np))

    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")
    if n_clusters > n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot be greater than n_samples ({n_samples})"
        )

    if random_state is not None:
        np.random.seed(random_state)

    use_gpu = (device.type == "cuda") and hasattr(faiss, "StandardGpuResources")

    kmeans = faiss.Kmeans(
        d,
        n_clusters,
        niter=niter,
        nredo=nredo,
        verbose=verbose,
        gpu=use_gpu,
        seed=random_state if random_state is not None else np.random.randint(2**31),
    )

    if device.type == "cuda" and not use_gpu:
        warnings.warn(
            "[TorchDR] WARNING: GPU device specified but faiss-gpu not installed. "
            "Using CPU for K-means. For GPU support, install faiss-gpu.",
            stacklevel=2,
        )

    kmeans.train(X_np)

    _, predicted_labels_np = kmeans.index.search(X_np, 1)
    predicted_labels_np = predicted_labels_np.ravel()

    predicted_labels_torch = torch.from_numpy(predicted_labels_np).long().to(device)
    labels_torch = labels.long().to(device)

    ari_metric = AdjustedRandScore().to(device)
    ari_score = ari_metric(predicted_labels_torch, labels_torch)

    if input_is_numpy:
        ari_score = ari_score.detach().cpu().numpy().item()
        predicted_labels = predicted_labels_np
    else:
        predicted_labels = predicted_labels_torch

    return ari_score, predicted_labels
