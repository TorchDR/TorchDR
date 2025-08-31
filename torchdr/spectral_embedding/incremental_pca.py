"""Incremental Principal Components Analysis module."""

# Authors: @sirluk
#
# License: BSD 3-Clause License

from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch

from torchdr.base import DRModule
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

        new_sample_count = torch.tensor([X.shape[0]], device=X.device)
        updated_sample_count = last_sample_count + new_sample_count

        if last_mean is None:
            last_sum = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
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
            # Store on computation device (either self.device or X.device if auto)
            param_device = X.device if self.device == "auto" else self.device
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

        if first_pass:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                (self.n_samples_seen_ / n_total_samples) * n_samples
            )
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)
            X = torch.vstack(
                (
                    self.singular_values_.view((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        U, S, Vt = self._svd_fn(X)
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
        X : ArrayLike of shape (n_samples, n_features)
            Data on which to fit the PCA model and project onto the components.
            Should fit in CPU memory but may be too large for GPU memory.
        y : Optional[Any], default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        ArrayLike of shape (n_samples, n_components)
            Projected data. Will be on the same device/backend as input.
        """
        # Track original format for output conversion
        X_, input_backend, input_device = to_torch(X, return_backend_device=True)

        # Validate the tensor
        X_ = self._validate_data(X_)
        n_samples, n_features = X_.shape
        if self.batch_size is None:
            self.batch_size = 5 * n_features

        # Determine computation device
        compute_device = X_.device if self.device == "auto" else self.device

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
