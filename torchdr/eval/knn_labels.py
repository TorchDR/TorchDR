"""K-NN label accuracy metric for evaluating embedding quality."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import numpy as np
import torch
import torch.distributed as dist
from typing import Union, Optional

from torchdr.utils import to_torch
from torchdr.distributed import DistributedContext
from torchdr.distance import pairwise_distances, FaissConfig


def knn_label_accuracy(
    X: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    k: int = 10,
    metric: str = "euclidean",
    backend: Optional[Union[str, FaissConfig]] = "faiss",
    exclude_self: bool = True,
    distributed: Union[bool, str] = "auto",
    return_per_sample: bool = False,
    device: Optional[str] = None,
):
    r"""Compute k-NN label accuracy to evaluate class structure preservation.

    This metric measures how well local class structure is preserved in the data
    representation (original or embedded space). For each point, it computes the
    proportion of its k-nearest neighbors that share the same class label.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
        Data representation (can be original features or embeddings).
    labels : torch.Tensor or np.ndarray of shape (n_samples,)
        Class labels for each sample. Can be integers or any comparable type.
    k : int, default=10
        Number of nearest neighbors to consider for each point.
    metric : str, default='euclidean'
        Distance metric to use for computing nearest neighbors.
        Options: 'euclidean', 'sqeuclidean', 'manhattan', 'angular'.
    backend : {'keops', 'faiss', None} or FaissConfig, default='faiss'
        Backend to use for k-NN computation:
        - 'keops': Memory-efficient symbolic computations
        - 'faiss': Fast approximate nearest neighbors (recommended for large datasets)
        - None: Standard PyTorch operations
        - FaissConfig object: FAISS with custom configuration
    exclude_self : bool, default=True
        Whether to exclude the point itself from its neighborhood.
        Usually True when evaluating on the same dataset used for k-NN search.
    distributed : bool or 'auto', default='auto'
        Whether to use multi-GPU distributed computation.
        - 'auto': Automatically detects if torch.distributed is initialized
        - True: Forces distributed mode (requires torch.distributed to be initialized)
        - False: Disables distributed mode
        When enabled:
        - Each GPU computes accuracy for its assigned chunk of samples
        - Device is automatically set to the local GPU rank
        - Backend is forced to 'faiss' for efficient distributed k-NN
        - Returns per-chunk results (no automatic gathering across GPUs)
        Requires launching with torchrun: ``torchrun --nproc_per_node=N script.py``
    return_per_sample : bool, default=False
        If True, returns per-sample accuracies instead of the mean.
        Shape: (n_samples,) or (chunk_size,) in distributed mode.
    device : str, optional
        Device to use for computation. If None, uses input device.

    Returns
    -------
    accuracy : float or torch.Tensor
        If return_per_sample=False: Mean k-NN label accuracy across all samples.
        If return_per_sample=True: Per-sample k-NN label accuracies.
        Value between 0 and 1, where 1 means all neighbors have same label.
        Returns numpy array/float if inputs are numpy, torch.Tensor otherwise.

    Examples
    --------
    >>> import torch
    >>> from torchdr.eval import knn_label_accuracy
    >>>
    >>> # Generate example data with 3 classes
    >>> X = torch.randn(300, 50)
    >>> labels = torch.repeat_interleave(torch.arange(3), 100)
    >>>
    >>> # Compute k-NN label accuracy
    >>> accuracy = knn_label_accuracy(X, labels, k=10)
    >>> print(f"k-NN label accuracy: {accuracy:.3f}")
    >>>
    >>> # Get per-sample accuracies
    >>> per_sample = knn_label_accuracy(X, labels, k=10, return_per_sample=True)
    >>> print(f"Mean: {per_sample.mean():.3f}, Std: {per_sample.std():.3f}")
    >>>
    >>> # Distributed computation (launch with: torchrun --nproc_per_node=4 script.py)
    >>> accuracy_chunk = knn_label_accuracy(X, labels, k=10, distributed=True)

    Notes
    -----
    This metric is useful for:
    - Evaluating how well embeddings preserve class structure
    - Comparing different DR methods on classification tasks
    - Assessing the quality of learned representations

    Higher values indicate better preservation of local class homogeneity.
    The metric is sensitive to class imbalance and noise in labels.

    In distributed mode, each GPU computes accuracy for its chunk of the dataset.
    To get the global accuracy, gather results from all GPUs and compute the mean.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    input_is_numpy = not isinstance(X, torch.Tensor) or not isinstance(
        labels, torch.Tensor
    )

    X = to_torch(X)
    labels = to_torch(labels)

    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"X and labels must have same number of samples, "
            f"got {X.shape[0]} and {labels.shape[0]}"
        )

    n_samples = X.shape[0]

    if k >= n_samples:
        raise ValueError(f"k ({k}) must be less than number of samples ({n_samples})")

    if distributed == "auto":
        distributed = dist.is_initialized()
    else:
        distributed = bool(distributed)

    if distributed:
        if not dist.is_initialized():
            raise RuntimeError(
                "[TorchDR] distributed=True requires launching with torchrun. "
                "Example: torchrun --nproc_per_node=4 your_script.py"
            )

        dist_ctx = DistributedContext()

        if device is None:
            device = X.device
        elif device == "cpu":
            raise ValueError(
                "[TorchDR] Distributed mode requires GPU (device cannot be 'cpu')"
            )

        device = torch.device(f"cuda:{dist_ctx.local_rank}")
    else:
        dist_ctx = None
        if device is None:
            device = X.device
        else:
            device = torch.device(device)

    X = X.to(device)
    labels = labels.to(device)

    _, neighbor_indices = pairwise_distances(
        X,
        metric=metric,
        backend=backend,
        k=k,
        exclude_diag=exclude_self,
        return_indices=True,
        device=device,
        distributed_ctx=dist_ctx,
    )

    neighbor_labels = labels[neighbor_indices]

    if dist_ctx is not None and dist_ctx.is_initialized:
        chunk_start, chunk_end = dist_ctx.compute_chunk_bounds(n_samples)
        query_labels = labels[chunk_start:chunk_end].unsqueeze(1)
    else:
        query_labels = labels.unsqueeze(1)

    matches = (neighbor_labels == query_labels).float()
    accuracies = matches.mean(dim=1)

    if return_per_sample:
        result = accuracies
        if input_is_numpy:
            result = result.detach().cpu().numpy()
    else:
        result = accuracies.mean()
        if input_is_numpy:
            result = result.detach().cpu().numpy().item()

    return result
