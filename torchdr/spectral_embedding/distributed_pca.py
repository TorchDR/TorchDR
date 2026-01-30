"""Distributed Principal Component Analysis module."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Optional, Union, Any

import numpy as np
import torch
import torch.distributed as dist

from torchdr.base import DRModule
from torchdr.utils import svd_flip
from torchdr.distributed import is_distributed, get_rank, get_world_size


class DistributedPCA(DRModule):
    r"""Distributed Principal Component Analysis for multi-GPU training.

    Uses the distributed covariance approach where each GPU:

    1. Computes local statistics (sum for mean, centered X.T @ X for covariance)
    2. Communicates via all-reduce (O(d²) communication)
    3. Computes final PCA on aggregated statistics

    This approach is communication-efficient when the number of samples is much
    larger than the number of features (n >> d), which is typical in
    dimensionality reduction.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to keep.
    verbose : bool, default=False
        Whether to print progress information.
    random_state : float, optional
        Random seed for reproducibility.

    Attributes
    ----------
    mean_ : torch.Tensor of shape (1, n_features)
        Per-feature empirical mean, estimated from the training set.
    components_ : torch.Tensor of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    embedding_ : torch.Tensor of shape (n_local_samples, n_components)
        The transformed local data chunk after calling fit_transform.

    Notes
    -----
    - Requires torch.distributed to be initialized (use torchrun or TorchDR CLI)
    - Automatically uses local_rank for GPU assignment
    - Each GPU only needs its data chunk in memory
    - For single-GPU workloads, use regular :class:`PCA` instead
    - Falls back to non-distributed mode if torch.distributed is not initialized

    Examples
    --------
    >>> # Script launched with: torchdr train.py --gpus 4
    >>> # or: torchrun --nproc_per_node=4 train.py
    >>> import torch
    >>> from torchdr import DistributedPCA
    >>>
    >>> # Each GPU loads its chunk of the data
    >>> rank = torch.distributed.get_rank()
    >>> world_size = torch.distributed.get_world_size()
    >>> chunk_size = len(full_data) // world_size
    >>> X_local = full_data[rank * chunk_size:(rank + 1) * chunk_size]
    >>>
    >>> # Distributed PCA - handles communication automatically
    >>> pca = DistributedPCA(n_components=50)
    >>> X_transformed = pca.fit_transform(X_local)
    >>>
    >>> # X_transformed contains this GPU's transformed chunk
    """

    def __init__(
        self,
        n_components: int = 2,
        verbose: bool = False,
        random_state: Optional[float] = None,
        **kwargs,
    ):
        # No device param - uses distributed context (local_rank)
        super().__init__(
            n_components=n_components,
            device="auto",  # Will use local GPU in distributed mode
            verbose=verbose,
            random_state=random_state,
            **kwargs,
        )
        self.mean_ = None
        self.components_ = None
        self._n_samples_total = None

    def _fit_transform(
        self, X_local: torch.Tensor, y: Optional[Any] = None
    ) -> torch.Tensor:
        """Fit the distributed PCA model and transform the local data chunk.

        Parameters
        ----------
        X_local : torch.Tensor of shape (n_local_samples, n_features)
            Local data chunk on this GPU/process.
        y : Optional[Any], default=None
            Ignored. Present for API compatibility.

        Returns
        -------
        embedding_ : torch.Tensor of shape (n_local_samples, n_components)
            Projected local data.
        """
        if not is_distributed():
            self.logger.warning(
                "torch.distributed is not initialized. Falling back to "
                "non-distributed PCA. Use regular PCA for single-GPU workloads."
            )
            return self._fit_transform_non_distributed(X_local)

        return self._fit_transform_distributed(X_local)

    def _fit_transform_non_distributed(self, X: torch.Tensor) -> torch.Tensor:
        """Fallback to regular PCA when distributed is not available."""
        original_device = X.device
        target_device = self._get_compute_device(X)
        if target_device != X.device:
            X_compute = X.to(target_device)
        else:
            X_compute = X

        self.mean_ = X_compute.mean(0, keepdim=True)
        X_centered = X_compute - self.mean_

        # SVD for PCA
        U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
        U, V = svd_flip(U, V)
        self.components_ = V[: self.n_components]
        self._n_samples_total = X.shape[0]

        self.embedding_ = U[:, : self.n_components] * S[: self.n_components]

        if original_device != X_compute.device:
            self.embedding_ = self.embedding_.to(original_device)

        return self.embedding_

    def _fit_transform_distributed(self, X_local: torch.Tensor) -> torch.Tensor:
        """Distributed PCA using all-reduce for covariance aggregation.

        Algorithm:
        1. All-reduce sum(X_local) and n_local to compute global mean
        2. Each GPU computes local covariance: (X_local - mean).T @ (X_local - mean)
        3. All-reduce covariance matrices to get global covariance
        4. Eigendecomposition on each GPU (or rank 0 + broadcast)
        5. Transform local data
        """
        original_device = X_local.device
        rank = get_rank()
        world_size = get_world_size()

        # Determine device - use local GPU if available
        local_rank = (
            int(torch.distributed.get_rank() % torch.cuda.device_count())
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else 0
        )

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            X_compute = X_local.to(device)
        else:
            device = X_local.device
            X_compute = X_local

        n_local = X_compute.shape[0]
        n_features = X_compute.shape[1]
        dtype = X_compute.dtype

        # Step 1: Compute global mean via all-reduce
        # Sum of samples on this GPU
        local_sum = X_compute.sum(dim=0)  # (n_features,)
        n_local_tensor = torch.tensor([n_local], dtype=dtype, device=device)

        # All-reduce to get global sum and count
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_local_tensor, op=dist.ReduceOp.SUM)

        n_total = int(n_local_tensor.item())
        self._n_samples_total = n_total
        self.mean_ = (local_sum / n_total).unsqueeze(0)  # (1, n_features)

        # Step 2: Compute local centered X.T @ X (covariance contribution)
        X_centered = X_compute - self.mean_
        local_cov = X_centered.T @ X_centered  # (n_features, n_features)

        # Step 3: All-reduce covariance matrices
        dist.all_reduce(local_cov, op=dist.ReduceOp.SUM)
        # Normalize to get covariance (we'll use n_total - 1 for unbiased estimate,
        # but for PCA this doesn't affect the eigenvectors)
        global_cov = local_cov / n_total

        # Step 4: Eigendecomposition
        # The covariance matrix C = X.T @ X / n has eigenvectors = right singular
        # vectors of X, and eigenvalues = singular values squared / n
        eigenvalues, eigenvectors = torch.linalg.eigh(global_cov)

        # eigh returns eigenvalues in ascending order, we want descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Apply sign flip for deterministic output (match sklearn convention)
        # Use the sign of the largest absolute value in each column
        max_abs_idx = torch.argmax(torch.abs(eigenvectors), dim=0)
        signs = torch.sign(
            eigenvectors[max_abs_idx, torch.arange(n_features, device=device)]
        )
        eigenvectors = eigenvectors * signs.unsqueeze(0)

        self.components_ = eigenvectors[
            :, : self.n_components
        ].T  # (n_components, n_features)

        # Step 5: Transform local data
        self.embedding_ = X_centered @ self.components_.T

        # Move back to original device if needed
        if original_device != device:
            self.embedding_ = self.embedding_.to(original_device)
            # Keep parameters on compute device for potential later use
            # but also store them on original device for transform()
            self.mean_ = self.mean_.to(original_device)
            self.components_ = self.components_.to(original_device)

        if self.verbose and rank == 0:
            self.logger.info(
                f"Distributed PCA: {n_total} samples across {world_size} GPUs, "
                f"reduced to {self.n_components} components."
            )

        return self.embedding_

    def transform(
        self, X: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        r"""Project input data onto the PCA components.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data to project onto the PCA components.

        Returns
        -------
        X_new : torch.Tensor or np.ndarray of shape (n_samples, n_components)
            Projected data.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError(
                "This DistributedPCA instance is not fitted yet. "
                "Call 'fit' or 'fit_transform' with some data first."
            )

        # Handle numpy input
        input_is_numpy = isinstance(X, np.ndarray)
        if input_is_numpy:
            X = torch.from_numpy(X)

        original_device = X.device
        if self.mean_.device != X.device:
            X_compute = X.to(self.mean_.device)
            result = (X_compute - self.mean_) @ self.components_.T
            result = result.to(original_device)
        else:
            result = (X - self.mean_) @ self.components_.T

        if input_is_numpy:
            return result.numpy()
        return result
