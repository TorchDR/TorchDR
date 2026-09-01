"""Principal Component Analysis module."""

# Authors: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import os
from typing import Optional, Union, Any

import numpy as np
import torch
import torch.distributed as dist

from torchdr.base import DRModule
from torchdr.utils import handle_input_output, svd_flip
from torchdr.distributed import is_distributed, get_rank, get_world_size


class PCA(DRModule):
    r"""Principal Component Analysis module.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to project the input data onto.
    device : str, default="auto"
        Device on which the computations are performed.
    distributed : str or bool, default="auto"
        Whether to use distributed mode for multi-GPU training.

        - "auto": Automatically detect if torch.distributed is initialized and
          use distributed mode if available.
        - True: Force distributed mode (requires torch.distributed to be initialized).
        - False: Disable distributed mode.

        In distributed mode, each GPU computes local statistics which are then
        aggregated using all-reduce operations. This is communication-efficient
        when the number of samples is much larger than the number of features
        (n >> d).
    verbose : bool, default=False
        Whether to print information during the computations.
    random_state : float, default=None
        Random seed for reproducibility.
    svd_driver : str, optional
        Name of the cuSOLVER method to be used for torch.linalg.svd.
        This keyword argument only works on CUDA inputs.
        Available options are: None, gesvd, gesvdj and gesvda.
        Defaults to None.

    Attributes
    ----------
    mean_ : torch.Tensor of shape (1, n_features)
        Per-feature empirical mean, calculated from the training set.
    components_ : torch.Tensor of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    embedding_ : torch.Tensor
        The transformed data after calling fit_transform.

    Examples
    --------
    Standard single-GPU usage::

        from torchdr import PCA
        import torch

        X = torch.randn(1000, 50)
        pca = PCA(n_components=10)
        X_reduced = pca.fit_transform(X)

    Multi-GPU distributed usage (launch with torchrun --nproc_per_node=4)::

        import torch
        from torchdr import PCA

        # Each GPU loads its chunk of the data
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        chunk_size = len(full_data) // world_size
        X_local = full_data[rank * chunk_size:(rank + 1) * chunk_size]

        # Distributed PCA - handles communication automatically
        pca = PCA(n_components=50, distributed="auto")
        X_transformed = pca.fit_transform(X_local)

    Notes
    -----
    In distributed mode:

    - Requires torch.distributed to be initialized (use torchrun or TorchDR CLI)
    - Automatically uses local_rank for GPU assignment
    - Each GPU only needs its data chunk in memory
    - Uses the covariance method: computes X.T @ X locally and aggregates via
      all-reduce, which has O(d^2) communication cost
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        distributed: Union[str, bool] = "auto",
        verbose: bool = False,
        random_state: float = None,
        svd_driver: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            verbose=verbose,
            random_state=random_state,
            **kwargs,
        )
        self.distributed = distributed
        self.svd_driver = svd_driver
        self.mean_ = None
        self.components_ = None
        self._n_samples_total = None

    def _should_use_distributed(self) -> bool:
        """Check if distributed mode should be used.

        Returns
        -------
        bool
            True if distributed mode should be used, False otherwise.
        """
        if self.distributed == "auto":
            return is_distributed()
        return bool(self.distributed)

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        """Fit the PCA model and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which to fit the PCA model and project onto the components.
        y : Optional[Any], default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        embedding_ : torch.Tensor of shape (n_samples, n_components)
            Projected data.
        """
        if self._should_use_distributed():
            return self._fit_transform_distributed(X)
        return self._fit_transform_standard(X)

    def _fit_transform_standard(self, X: torch.Tensor) -> torch.Tensor:
        """Standard single-GPU PCA implementation.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which to fit the PCA model and project onto the components.

        Returns
        -------
        embedding_ : torch.Tensor of shape (n_samples, n_components)
            Projected data.
        """
        original_device = X.device
        target_device = self._get_compute_device(X)
        if target_device != X.device:
            X_compute = X.to(target_device)
        else:
            X_compute = X

        self.mean_ = X_compute.mean(0, keepdim=True)
        U, S, V = torch.linalg.svd(
            X_compute - self.mean_, full_matrices=False, driver=self.svd_driver
        )
        U, V = svd_flip(U, V)  # flip eigenvectors' sign to enforce deterministic output
        self.components_ = V[: self.n_components]

        self.embedding_ = U[:, : self.n_components] * S[: self.n_components]

        # Move embedding back to original device, but keep parameters on compute device
        if original_device != X_compute.device:
            self.embedding_ = self.embedding_.to(original_device)

        return self.embedding_

    def _fit_transform_distributed(self, X_local: torch.Tensor) -> torch.Tensor:
        """Distributed PCA using all-reduce for covariance aggregation.

        Algorithm:
        1. All-reduce sum(X_local) and n_local to compute global mean
        2. Each GPU computes local covariance: (X_local - mean).T @ (X_local - mean)
        3. All-reduce covariance matrices to get global covariance
        4. Eigendecomposition on rank 0, broadcast components to all GPUs
        5. Transform local data

        Parameters
        ----------
        X_local : torch.Tensor of shape (n_local_samples, n_features)
            Local data chunk on this GPU/process.

        Returns
        -------
        embedding_ : torch.Tensor of shape (n_local_samples, n_components)
            Projected local data.
        """
        if not is_distributed():
            self.logger.warning(
                "torch.distributed is not initialized but distributed=True. "
                "Falling back to standard PCA."
            )
            return self._fit_transform_standard(X_local)

        original_device = X_local.device
        rank = get_rank()
        world_size = get_world_size()

        # Determine device - use local GPU if available (LOCAL_RANK is set by torchrun)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
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
        # Normalize to get covariance
        global_cov = local_cov / n_total

        # Step 4: Eigendecomposition on rank 0, then broadcast
        # Computing on one GPU and broadcasting guarantees identical components
        # across all GPUs (avoids potential non-determinism in eigh)
        self.components_ = torch.empty(
            self.n_components, n_features, dtype=dtype, device=device
        )

        if rank == 0:
            eigenvalues, eigenvectors = torch.linalg.eigh(global_cov)

            # eigh returns eigenvalues in ascending order, we want descending
            idx = torch.argsort(eigenvalues, descending=True)
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
            ].T.contiguous()  # (n_components, n_features)

        dist.broadcast(self.components_, src=0)

        # Step 5: Transform local data
        self.embedding_ = X_centered @ self.components_.T

        # Move back to original device if needed
        if original_device != device:
            self.embedding_ = self.embedding_.to(original_device)
            # Keep parameters on original device for transform()
            self.mean_ = self.mean_.to(original_device)
            self.components_ = self.components_.to(original_device)

        if self.verbose and rank == 0:
            self.logger.info(
                f"Distributed PCA: {n_total} samples across {world_size} GPUs, "
                f"reduced to {self.n_components} components."
            )

        return self.embedding_

    @handle_input_output()
    def transform(
        self, X: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        r"""Project the input data onto the PCA components.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data to project onto the PCA components.

        Returns
        -------
        X_new : torch.Tensor or np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
        if self.mean_.device != X.device:
            X_compute = X.to(self.mean_.device)
            result = (X_compute - self.mean_) @ self.components_.T
            return result.to(X.device)
        else:
            return (X - self.mean_) @ self.components_.T
