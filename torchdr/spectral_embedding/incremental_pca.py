"""Incremental Principal Components Analysis module."""

# Authors: @sirluk
#
# License: BSD 3-Clause License

from functools import partial
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torchdr.base import DRModule
from torchdr.distributed import is_distributed, get_rank, get_world_size
from torchdr.utils import (
    handle_input_output,
    svd_flip,
    to_torch,
    restore_original_format,
    validate_tensor,
)

from typing import Any, TypeVar

ArrayLike = TypeVar("ArrayLike", torch.Tensor, np.ndarray)


class IncrementalPCA(DRModule):
    """Incremental Principal Components Analysis (IPCA) leveraging PyTorch for GPU acceleration.

    This class provides methods to fit the model on data incrementally in batches,
    and to transform new data based on the principal components learned during the fitting process.

    The algorithm uses incremental SVD updates based on :cite:p:`ross2008incremental`,
    which allows maintaining a low-rank approximation of the data covariance matrix
    without storing all data or recomputing from scratch.

    It is particularly useful when the dataset to be decomposed is too large to fit in memory.
    Adapted from `Scikit-learn Incremental PCA <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py>`_.

    Memory Management Strategy:
    - Data is processed in batches to avoid loading entire dataset into memory
    - Each batch is temporarily moved to computation device (GPU if specified)
    - Model parameters (mean_, components_) are kept on computation device
    - Only the current batch needs to fit in GPU memory, not the full dataset

    Examples
    --------
    Using with PyTorch DataLoader for true out-of-core learning::

        from torch.utils.data import DataLoader, TensorDataset

        # Create a DataLoader for a huge dataset
        dataset = TensorDataset(huge_X_tensor, huge_y_tensor)
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

        # Fit incrementally using DataLoader
        ipca = IncrementalPCA(n_components=50, device='cuda')
        for batch in dataloader:
            X_batch = batch[0]  # DataLoader returns (X, y) tuples
            ipca.partial_fit(X_batch)

        # Transform new data in batches
        test_loader = DataLoader(test_dataset, batch_size=1000)
        for batch in test_loader:
            X_batch = batch[0]
            X_transformed = ipca.transform(X_batch)
            # Process transformed batch...

    Using with data generators for streaming large files::

        import pandas as pd

        def data_generator():
            # Read huge CSV in chunks
            for chunk in pd.read_csv('huge_file.csv', chunksize=1000):
                yield torch.tensor(chunk.values, dtype=torch.float32)

        ipca = IncrementalPCA(n_components=50)
        for batch in data_generator():
            ipca.partial_fit(batch)

    Using with HDF5 or memory-mapped arrays::

        import h5py

        with h5py.File('huge_dataset.h5', 'r') as f:
            X = f['data']  # HDF5 dataset, not loaded into memory
            n_samples = X.shape[0]
            batch_size = 1000

            ipca = IncrementalPCA(n_components=100)
            for i in range(0, n_samples, batch_size):
                batch = torch.tensor(X[i:i+batch_size])
                ipca.partial_fit(batch)

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If `None`, it's set to the minimum of the
        number of samples and features. Defaults to None.
    copy : bool
        If False, input data will be overwritten. Defaults to True.
    batch_size : int, optional
        The number of samples to use for each batch. Only needed if self.fit is called.
        If `None`, it's inferred from the data and set to `5 * n_features`.
        Defaults to None.
    svd_driver : str, optional
        Name of the cuSOLVER method to be used for torch.linalg.svd.
        This keyword argument only works on CUDA inputs.
        Available options are: None, gesvd, gesvdj and gesvda.
        Defaults to None.
    lowrank : bool, optional
        Whether to use torch.svd_lowrank instead of torch.linalg.svd which can be faster.
        Defaults to False.
    lowrank_q : int, optional
        For an adequate approximation of n_components, this parameter defaults to
        n_components * 2.
    lowrank_niter : int, optional
        Number of subspace iterations to conduct for torch.svd_lowrank.
        Defaults to 4.
    device : str, optional
        Device on which the computations are performed. Defaults to "auto".
    random_state : float, optional
        Random state for reproducibility. Defaults to None.
    **kwargs : dict
        Additional keyword arguments.
    """  # noqa: E501

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: Optional[bool] = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4,
        device: str = "auto",
        verbose: bool = False,
        random_state: float = None,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            verbose=verbose,
            random_state=random_state,
            **kwargs,
        )
        self.copy = copy
        self.batch_size = batch_size
        self.n_features_ = None
        self.svd_driver = svd_driver
        self.lowrank = lowrank
        self.lowrank_q = lowrank_q
        self.lowrank_niter = lowrank_niter

        if lowrank:
            if lowrank_q is None:
                assert n_components is not None, (
                    "n_components must be specified when using lowrank mode "
                )
                "with lowrank_q=None."
                lowrank_q = n_components * 2
            assert lowrank_q >= n_components, (
                "lowrank_q must be greater than or equal to n_components."
            )

            def svd_fn(X):
                U, S, V = torch.svd_lowrank(X, q=lowrank_q, niter=lowrank_niter)
                return U, S, V.mH  # V is returned as a conjugate transpose

            self._svd_fn = svd_fn

        else:
            self._svd_fn = partial(
                torch.linalg.svd, full_matrices=False, driver=svd_driver
            )

    def _validate_data(self, X) -> torch.Tensor:
        """Validate and prepare input data for processing.

        Ensures the input is a torch tensor with appropriate dtype and shape.
        Does NOT move data to computation device - that happens per batch.

        Parameters
        ----------
        X : array-like
            Input data to validate.

        Returns
        -------
        torch.Tensor
            Validated tensor on its original device.
        """
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = to_torch(X)
        elif self.copy:
            X = X.clone()

        # Use standard validation
        X = validate_tensor(
            X,
            ensure_2d=True,
            ensure_min_samples=1,
            ensure_min_features=1,
            max_components=self.n_components,
        )

        # IncrementalPCA-specific validation for batch size
        n_samples = X.shape[0]
        if self.n_components is not None and self.n_components > n_samples:
            raise ValueError(
                f"n_components={self.n_components} must be less or equal "
                f"to the batch number of samples {n_samples}."
            )

        # Ensure float dtype (validate_tensor doesn't handle dtype conversion)
        if X.dtype not in [torch.float32, torch.float64]:
            X = X.to(torch.float32)

        return X

    @staticmethod
    def _incremental_mean_and_var(
        X, last_mean, last_variance, last_sample_count
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incrementally compute mean and variance using Welford's algorithm.

        This method implements a numerically stable algorithm for computing running
        statistics without needing access to all previous data. It works by:

        1. Computing the sum of the new batch in float64 for precision
        2. Combining it with the previous sum (mean * count) from past batches
        3. Using Welford's update formula to compute variance, which involves:
           - Computing deviation of each point from the batch mean
           - Applying a correction term to avoid numerical cancellation
           - Combining with previous variance using weighted averaging

        The algorithm uses float64 internally because:
        - Sums of millions of float32 values accumulate significant rounding error
        - The variance formula involves subtracting large similar numbers (catastrophic cancellation)
        - Each batch compounds any numerical errors from previous batches

        Parameters
        ----------
        X : torch.Tensor
            Current batch of data (n_samples, n_features).
        last_mean : torch.Tensor or None
            Mean from previous batches (n_features,). None on first batch.
        last_variance : torch.Tensor or None
            Variance from previous batches (n_features,). None on first batch.
        last_sample_count : torch.Tensor
            Number of samples seen in previous batches. 0 on first batch.

        Returns
        -------
        updated_mean : torch.Tensor
            Updated mean including current batch (n_features,).
        updated_variance : torch.Tensor
            Updated variance including current batch (n_features,).
        updated_sample_count : torch.Tensor
            Total number of samples seen (scalar).
        """
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        if last_sample_count > 0:
            assert last_mean is not None, (
                "last_mean should not be None if last_sample_count > 0."
            )
            assert last_variance is not None, (
                "last_variance should not be None if last_sample_count > 0."
            )

        # Use device of last_sample_count which is always provided and on the right device
        device = last_sample_count.device
        new_sample_count = torch.tensor([X.shape[0]], device=device)
        updated_sample_count = last_sample_count + new_sample_count

        if last_mean is None:
            last_sum = torch.zeros(X.shape[1], dtype=torch.float64, device=device)
        else:
            last_sum = last_mean * last_sample_count

        new_sum = X.sum(dim=0, dtype=torch.float64)

        updated_mean = (last_sum + new_sum) / updated_sample_count

        T = new_sum / new_sample_count
        temp = X - T
        correction = temp.sum(dim=0, dtype=torch.float64).square()
        temp.square_()
        new_unnormalized_variance = temp.sum(dim=0, dtype=torch.float64)
        new_unnormalized_variance -= correction / new_sample_count
        if last_variance is None:
            updated_variance = new_unnormalized_variance / updated_sample_count
        else:
            last_unnormalized_variance = last_variance * last_sample_count
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum).square()
            )
            updated_variance = updated_unnormalized_variance / updated_sample_count

        return updated_mean, updated_variance, updated_sample_count

    def partial_fit(self, X, check_input=True):
        """Fit incrementally the model with batch data `X`.

        This method updates the PCA model with a new batch of data without
        requiring access to previously seen data. It maintains running statistics
        (mean, variance) and incrementally updates the principal components.

        The batch X should already be on the computation device when called
        from _fit_transform. When called directly, X can be on any device.

        Parameters
        ----------
        X : torch.Tensor
            The batch input data tensor with shape (n_samples, n_features).
            Should fit in memory/GPU memory.
        check_input : bool, optional
            If True, validates the input. Defaults to True.

        Returns
        -------
        IncrementalPCA:
            The updated IPCA model after processing the batch.

        Examples
        --------
        Basic usage with manual batching::

            ipca = IncrementalPCA(n_components=10)
            for i in range(0, len(X), batch_size):
                ipca.partial_fit(X[i:i+batch_size])

        With PyTorch DataLoader (recommended for large datasets)::

            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=256)

            ipca = IncrementalPCA(n_components=50)
            for batch in dataloader:
                # Handle DataLoader's (X, y) tuple format
                X_batch = batch[0] if isinstance(batch, tuple) else batch
                ipca.partial_fit(X_batch)

        Notes
        -----
        - Parameters (mean_, components_, etc.) are stored on the computation device,
          which is either self.device (if specified) or the device of the first batch
          (if self.device == "auto").
        - Uses Welford's algorithm for numerically stable incremental mean/variance.
        - SVD is performed on augmented matrix containing previous components
          and current batch to update the decomposition.
        - This is the recommended method for fitting with DataLoader or generators.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        # Initialize attributes to avoid errors during the first call to partial_fit
        if first_pass:
            self.mean_ = None
            self.var_ = None
            # Store on computation device
            param_device = self._get_compute_device(X)
            self.n_samples_seen_ = torch.tensor([0], device=param_device)
            self.n_features_ = n_features
            if not self.n_components:
                self.n_components = min(n_samples, n_features)

        if n_features != self.n_features_:
            raise ValueError(
                f"n_features={self.n_features_} while input has {n_features} features"
            )

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, self.n_samples_seen_
        )

        # Center the data without modifying input in-place (sklearn-style)
        if first_pass:
            X_centered = X - col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X_centered = X - col_batch_mean
            mean_correction_factor = torch.sqrt(
                (self.n_samples_seen_ / n_total_samples) * n_samples
            )
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)
            X_centered = torch.vstack(
                (
                    self.singular_values_.view((-1, 1)) * self.components_,
                    X_centered,
                    mean_correction,
                )
            )

        U, S, Vt = self._svd_fn(X_centered)
        U, Vt = svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[: self.n_components]
        self.singular_values_ = S[: self.n_components]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]
        if self.n_components not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components :].mean()
        else:
            # Use same device as other parameters
            self.noise_variance_ = torch.tensor(0.0, device=self.mean_.device)
        return self

    @handle_input_output()
    def transform(self, X: ArrayLike) -> ArrayLike:
        """Apply dimensionality reduction to `X`.

        Projects input data onto the principal components learned during fitting.
        Unlike fit, this processes the full input at once, not in batches.

        Device Management:
        - If X and parameters are on different devices, X is temporarily moved
          to parameters' device for computation
        - Result is moved back to X's original device
        - This avoids moving parameters which would be inefficient for repeated transforms

        Parameters
        ----------
        X : ArrayLike
            New data with shape (n_samples, n_features) to be transformed.
            Can be on any device.

        Returns
        -------
        ArrayLike
            Transformed data with shape (n_samples, n_components).
            Will be on the same device and format as input X.
        """
        # Move input to the same device as the model parameters for computation
        if self.mean_.device != X.device:
            X_compute = X.to(self.mean_.device)
            # Ensure dtype compatibility after moving
            mean = (
                self.mean_.to(X_compute.dtype)
                if self.mean_.dtype != X_compute.dtype
                else self.mean_
            )
            components = (
                self.components_.to(X_compute.dtype)
                if self.components_.dtype != X_compute.dtype
                else self.components_
            )
            result = (X_compute - mean) @ components.T
            # Move result back to original device
            return result.to(X.device)
        else:
            # Ensure dtype compatibility
            mean = self.mean_.to(X.dtype) if self.mean_.dtype != X.dtype else self.mean_
            components = (
                self.components_.to(X.dtype)
                if self.components_.dtype != X.dtype
                else self.components_
            )
            return (X - mean) @ components.T

    def _fit_transform(self, X: ArrayLike, y: Optional[Any] = None) -> ArrayLike:
        """Fit the model with X and apply the dimensionality reduction on X.

        This is the main entry point that orchestrates batch processing:
        1. Splits data into batches (only batches need to fit in GPU memory)
        2. Moves each batch to computation device
        3. Calls partial_fit to incrementally update the model
        4. After fitting all batches, transforms the full dataset

        Memory Strategy:
        - Full dataset X stays on its original device (typically CPU)
        - Only individual batches are moved to GPU for processing
        - This allows processing datasets larger than GPU memory
        - Final transform operates on full dataset (already in memory)

        Parameters
        ----------
        X : ArrayLike or DataLoader
            Data on which to fit the PCA model and project onto the components.
            Can be a tensor/array (should fit in CPU memory) or a DataLoader
            for streaming large datasets. DataLoader batches can be tuples (X, y).
        y : Optional[Any], default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        ArrayLike of shape (n_samples, n_components)
            Projected data. Will be on the same device/backend as input.
        """
        # Handle DataLoader input
        if isinstance(X, DataLoader):
            # Single pass: fit incrementally
            for batch in X:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                # Move to explicit device if specified, otherwise keep on native device
                if self.device != "auto":
                    batch = batch.to(self.device)
                self.partial_fit(batch, check_input=True)

            # Second pass: transform all data
            embeddings = []
            for batch in X:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                embeddings.append(self.transform(batch))
            self.embedding_ = torch.cat(embeddings, dim=0)
            return self.embedding_

        # Track original format for output conversion
        X_, input_backend, input_device = to_torch(X, return_backend_device=True)

        # Validate the tensor
        X_ = self._validate_data(X_)
        n_samples, n_features = X_.shape
        if self.batch_size is None:
            self.batch_size = 5 * n_features

        # Determine computation device
        compute_device = self._get_compute_device(X_)

        for batch in self.gen_batches(
            n_samples, self.batch_size, min_batch_size=self.n_components or 0
        ):
            # Move batch to computation device if needed
            X_batch = X_[batch]
            if X_batch.device != compute_device:
                X_batch = X_batch.to(compute_device)
            self.partial_fit(X_batch, check_input=False)

        # Transform and convert back to original format
        self.embedding_ = self.transform(X_)
        self.embedding_ = restore_original_format(
            self.embedding_, backend=input_backend, device=input_device
        )
        return self.embedding_

    @staticmethod
    def gen_batches(n: int, batch_size: int, min_batch_size: int = 0):
        """Generate slices containing `batch_size` elements from 0 to `n`.

        Used to split the dataset into manageable batches that fit in GPU memory.
        The last slice may contain less than `batch_size` elements, when
        `batch_size` does not divide `n`.

        Parameters
        ----------
        n : int
            Total size of the dataset.
        batch_size : int
            Number of samples in each batch. Should be chosen to fit in GPU memory.
        min_batch_size : int, optional
            Minimum number of samples in each batch. Used to ensure batches
            have enough samples for meaningful statistics. Defaults to 0.

        Yields
        ------
        slice:
            A slice object representing indices [start:end] for the current batch.

        Examples
        --------
        >>> list(IncrementalPCA.gen_batches(10, 3))
        [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 10)]
        """
        start = 0
        for _ in range(int(n // batch_size)):
            end = start + batch_size
            if end + min_batch_size > n:
                continue
            yield slice(start, end)
            start = end
        if start < n:
            yield slice(start, n)


class ExactIncrementalPCA(DRModule):
    r"""Exact Incremental Principal Component Analysis.

    This implementation computes the exact PCA solution by incrementally building
    the covariance matrix X.T @ X in batches. This is memory-efficient when the
    number of features is small, as only the (n_features, n_features) covariance
    matrix needs to be stored, not the full dataset.

    Unlike IncrementalPCA which uses an approximate incremental SVD algorithm,
    this method computes the exact PCA solution but requires two passes through
    the data: one to compute the mean, and one to build the covariance matrix.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
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

    Attributes
    ----------
    mean_ : torch.Tensor of shape (n_features,)
        Per-feature empirical mean, calculated from the training set.
    components_ : torch.Tensor of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    explained_variance_ : torch.Tensor of shape (n_components,)
        The amount of variance explained by each of the selected components.
    n_samples_seen_ : int
        The number of samples processed.
    n_features_in_ : int
        Number of features seen during fit.

    Notes
    -----
    When to use each incremental PCA variant:

    - **IncrementalPCA**: Use when you need single-pass processing, can tolerate
      approximate results, or have high-dimensional data where storing the full
      covariance matrix would be prohibitive.

    - **ExactIncrementalPCA**: Use when you need exact PCA results, have
      low-dimensional data (small n_features), and can afford two passes
      through the data.

    In distributed mode:

    - Requires torch.distributed to be initialized (use torchrun or TorchDR CLI)
    - Automatically uses local_rank for GPU assignment
    - Each GPU only needs its data chunk in memory
    - Uses covariance aggregation: O(d) communication for mean, O(d^2) for covariance
    - Mathematically equivalent to running on concatenated data from all GPUs

    Examples
    --------
    Using with PyTorch DataLoader for large datasets::

        from torch.utils.data import DataLoader, TensorDataset
        from torchdr.spectral_embedding import ExactIncrementalPCA

        # Create a DataLoader for a huge dataset
        dataset = TensorDataset(huge_X_tensor)
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

        # Initialize the model
        pca = ExactIncrementalPCA(n_components=50, device='cuda')

        # First pass: compute mean
        batch_list = []
        for batch in dataloader:
            X_batch = batch[0]  # DataLoader returns tuples
            batch_list.append(X_batch)
        pca.compute_mean(batch_list)

        # Second pass: fit the model
        pca.fit(batch_list)

        # Transform new data
        test_loader = DataLoader(test_dataset, batch_size=1000)
        transformed_batches = []
        for batch in test_loader:
            X_batch = batch[0]
            X_transformed = pca.transform(X_batch)
            transformed_batches.append(X_transformed)

    Using with data generators for streaming::

        import torch
        from torchdr.spectral_embedding import ExactIncrementalPCA

        # Generate large dataset that doesn't fit in memory
        def data_generator():
            for i in range(100):  # 100 batches
                yield torch.randn(1000, 50)  # 1000 samples, 50 features

        # First pass: compute mean
        pca = ExactIncrementalPCA(n_components=10)
        pca.compute_mean(data_generator())

        # Second pass: fit the model
        pca.fit(data_generator())

        # Transform new data
        X_new = torch.randn(100, 50)
        X_transformed = pca.transform(X_new)

    Multi-GPU distributed usage (launch with torchrun --nproc_per_node=4)::

        import torch
        from torchdr import ExactIncrementalPCA

        # Each GPU loads its chunk of the data
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        chunk_size = len(full_data) // world_size
        X_local = full_data[rank * chunk_size:(rank + 1) * chunk_size]

        # Create batches for incremental processing
        batch_size = 1000
        batches = [X_local[i:i+batch_size] for i in range(0, len(X_local), batch_size)]

        # Distributed PCA - handles communication automatically
        pca = ExactIncrementalPCA(n_components=50, distributed="auto")
        pca.compute_mean(batches)  # First pass: compute global mean
        pca.fit(batches)           # Second pass: build global covariance
        X_transformed = pca.transform(X_local)  # Transform local data
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        distributed: Union[str, bool] = "auto",
        verbose: bool = False,
        random_state: float = None,
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
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.n_samples_seen_ = 0
        self.n_features_in_ = None
        self._XtX = None  # Accumulated X.T @ X matrix

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

    def _get_device(self):
        """Get the device for this rank in distributed mode."""
        if self._should_use_distributed() and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def compute_mean(self, X_batches):
        """Compute the mean from batches of data (first pass).

        In distributed mode, each GPU computes its local sum and sample count,
        then all-reduce is used to compute the global mean.

        Parameters
        ----------
        X_batches : iterable of torch.Tensor or single torch.Tensor
            Either an iterable yielding batches of data, a DataLoader, or a single tensor.
            Each batch should have shape (n_samples, n_features).
            DataLoader batches can be tuples (X, y) - only X will be used.

        Returns
        -------
        self : ExactIncrementalPCA
            Returns the instance itself.
        """
        if self._should_use_distributed():
            return self._compute_mean_distributed(X_batches)
        return self._compute_mean_standard(X_batches)

    def _compute_mean_standard(self, X_batches):
        """Standard (non-distributed) mean computation."""
        total_sum = None
        n_samples_seen = 0

        # Handle single tensor input
        if isinstance(X_batches, (torch.Tensor, np.ndarray)):
            X_batches = [X_batches]

        for batch in X_batches:
            # Handle DataLoader tuple batches (X, y)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = to_torch(batch)

            if self.device != "auto" and self.device is not None:
                batch = batch.to(self.device)

            if total_sum is None:
                self.n_features_in_ = batch.shape[1]
                total_sum = torch.zeros(
                    self.n_features_in_, dtype=batch.dtype, device=batch.device
                )

            total_sum += batch.sum(dim=0)
            n_samples_seen += batch.shape[0]

        self.n_samples_seen_ = n_samples_seen
        self.mean_ = total_sum / n_samples_seen

        if self.verbose:
            self.logger.info(f"Computed mean from {n_samples_seen} samples")

        return self

    def _compute_mean_distributed(self, X_batches):
        """Distributed mean computation using all-reduce."""
        if not is_distributed():
            self.logger.warning(
                "torch.distributed is not initialized but distributed=True. "
                "Falling back to standard compute_mean."
            )
            return self._compute_mean_standard(X_batches)

        device = self._get_device()
        local_sum = None
        n_local = 0

        # Handle single tensor input
        if isinstance(X_batches, (torch.Tensor, np.ndarray)):
            X_batches = [X_batches]

        for batch in X_batches:
            # Handle DataLoader tuple batches (X, y)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = to_torch(batch).to(device)

            if local_sum is None:
                self.n_features_in_ = batch.shape[1]
                local_sum = torch.zeros(
                    self.n_features_in_, dtype=batch.dtype, device=device
                )

            local_sum += batch.sum(dim=0)
            n_local += batch.shape[0]

        # All-reduce to get global sum and count
        n_local_tensor = torch.tensor([n_local], dtype=local_sum.dtype, device=device)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_local_tensor, op=dist.ReduceOp.SUM)

        n_total = int(n_local_tensor.item())
        self.n_samples_seen_ = n_total
        self.mean_ = local_sum / n_total

        if self.verbose and get_rank() == 0:
            self.logger.info(
                f"Computed global mean from {n_total} samples across "
                f"{get_world_size()} GPUs"
            )

        return self

    def partial_fit(self, X: torch.Tensor):
        """Incrementally fit the model with a batch of samples.

        This method assumes the mean has already been computed using compute_mean().
        Accumulates X_centered.T @ X_centered for one batch.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training batch.

        Returns
        -------
        self : ExactIncrementalPCA
            Returns the instance itself.
        """
        if self.mean_ is None:
            raise ValueError("Mean must be computed first using compute_mean()")

        if self._should_use_distributed():
            return self._partial_fit_distributed(X)
        return self._partial_fit_standard(X)

    def _partial_fit_standard(self, X: torch.Tensor):
        """Standard (non-distributed) partial fit."""
        # Handle DataLoader tuple batches (X, y)
        if isinstance(X, (list, tuple)):
            X = X[0]
        X = to_torch(X)

        if self.device != "auto" and self.device is not None:
            X = X.to(self.device)

        # Center the data
        X_centered = X - self.mean_

        # Initialize or update X.T @ X
        if self._XtX is None:
            self._XtX = torch.zeros(
                (self.n_features_in_, self.n_features_in_),
                dtype=X.dtype,
                device=self._get_compute_device(X),
            )

        self._XtX += X_centered.T @ X_centered

        return self

    def _partial_fit_distributed(self, X: torch.Tensor):
        """Distributed partial fit - accumulates local covariance."""
        if not is_distributed():
            return self._partial_fit_standard(X)

        device = self._get_device()

        # Handle DataLoader tuple batches (X, y)
        if isinstance(X, (list, tuple)):
            X = X[0]
        X = to_torch(X).to(device)

        # Center the data using global mean
        X_centered = X - self.mean_

        # Initialize or update X.T @ X
        if self._XtX is None:
            self._XtX = torch.zeros(
                (self.n_features_in_, self.n_features_in_),
                dtype=X.dtype,
                device=device,
            )

        self._XtX += X_centered.T @ X_centered

        return self

    def fit(self, X_batches, y=None):
        """Fit the model with batches of samples.

        This method assumes the mean has already been computed using compute_mean().
        If mean is not computed, it will compute it first (requiring two passes).

        In distributed mode, each GPU computes its local covariance contribution,
        then all-reduce is used to compute the global covariance matrix before
        eigendecomposition.

        Parameters
        ----------
        X_batches : iterable of torch.Tensor, DataLoader, or single torch.Tensor
            Either an iterable yielding batches of data, a DataLoader, or a single
            tensor. Each batch should have shape (n_samples, n_features).
            DataLoader batches can be tuples (X, y) - only X will be used.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        self : ExactIncrementalPCA
            Returns the instance itself.
        """
        # Handle different input types
        if isinstance(X_batches, (torch.Tensor, np.ndarray)):
            X_batches_iter = [X_batches]
        elif isinstance(X_batches, DataLoader):
            # DataLoader can be re-iterated without storing
            X_batches_iter = X_batches
        elif hasattr(X_batches, "__iter__"):
            # For generators/iterables, convert to list to allow multiple passes
            X_batches_iter = list(X_batches)
        else:
            raise TypeError(
                f"[TorchDR] X_batches must be a tensor, array, DataLoader, or iterable. "
                f"Got {type(X_batches).__name__}."
            )

        # Compute mean if not already done
        if self.mean_ is None:
            should_log = self.verbose and (
                not self._should_use_distributed() or get_rank() == 0
            )
            if should_log:
                self.logger.info("Computing mean (first pass through data)")
            self.compute_mean(X_batches_iter)

        # Reset XtX for fresh computation
        self._XtX = None

        # Build covariance matrix (second pass)
        should_log = self.verbose and (
            not self._should_use_distributed() or get_rank() == 0
        )
        if should_log:
            self.logger.info("Building covariance matrix (second pass through data)")

        for batch in X_batches_iter:
            self.partial_fit(batch)

        # Compute eigendecomposition (with all-reduce in distributed mode)
        self._compute_components()

        return self

    def _compute_components(self):
        """Compute principal components from the accumulated covariance matrix.

        In distributed mode, performs all-reduce on the covariance matrix first.
        """
        if self._XtX is None:
            raise ValueError("No data has been fitted yet")

        # In distributed mode, all-reduce covariance matrices
        if self._should_use_distributed() and is_distributed():
            dist.all_reduce(self._XtX, op=dist.ReduceOp.SUM)

        # Compute covariance matrix
        covariance = self._XtX / self.n_samples_seen_

        # Compute eigendecomposition
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        # Select top n_components (reverse order for descending eigenvalues)
        n_components = min(self.n_components, self.n_features_in_)
        idx = torch.argsort(eigenvalues, descending=True)[:n_components]

        self.explained_variance_ = eigenvalues[idx]
        self.components_ = eigenvectors[:, idx].T

        # Apply deterministic sign flip (same convention as distributed PCA)
        # Use the sign of the largest absolute value in each row
        device = self.components_.device
        max_abs_idx = torch.argmax(torch.abs(self.components_), dim=1)
        signs = torch.sign(
            self.components_[torch.arange(n_components, device=device), max_abs_idx]
        )
        self.components_ = self.components_ * signs.unsqueeze(1)

        should_log = self.verbose and (
            not self._should_use_distributed() or get_rank() == 0
        )
        if should_log:
            if self._should_use_distributed():
                self.logger.info(
                    f"Computed {n_components} principal components from "
                    f"{self.n_samples_seen_} samples across {get_world_size()} GPUs"
                )
            else:
                self.logger.info(f"Computed {n_components} principal components")

    def _fit_transform(self, X_batches, y=None):
        """Internal fit_transform without decorator.

        Parameters
        ----------
        X_batches : torch.Tensor
            Single tensor of data to fit and transform.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        X_transformed : torch.Tensor
            Transformed data.
        """
        # For a single tensor, we can fit and transform efficiently
        self.compute_mean([X_batches])
        self.fit([X_batches])
        return self.transform(X_batches)

    @handle_input_output
    def fit_transform(self, X_batches, y=None):
        """Fit the model and transform the data.

        Parameters
        ----------
        X_batches : iterable of torch.Tensor or single torch.Tensor
            Either an iterable yielding batches of data, or a single tensor.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        X_transformed : torch.Tensor
            Transformed data (concatenated from all batches).
        """
        # Fit the model
        self.fit(X_batches, y)

        # Transform the data
        # Need to re-iterate through the data
        if isinstance(X_batches, (torch.Tensor, np.ndarray)):
            return self.transform(X_batches)
        else:
            # For generator input, we'd need to store data or require another pass
            raise NotImplementedError(
                "fit_transform with generator input requires storing data. "
                "Please use fit() followed by transform() on your data."
            )

    @handle_input_output
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply dimensionality reduction on X.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : torch.Tensor of shape (n_samples, n_components)
            Transformed data.
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet")

        if self.device != "auto" and self.device is not None:
            X = X.to(self.device)

        # Center and project
        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T

        return X_transformed
