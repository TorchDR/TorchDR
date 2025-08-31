"""Exact Incremental Principal Component Analysis module."""

# Authors: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import numpy as np
import torch

from torchdr.base import DRModule
from torchdr.utils import handle_input_output, to_torch, set_logger


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
    """

    def __init__(
        self,
        n_components: int = 2,
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
        self.logger = set_logger(self.__class__.__name__, verbose)
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.n_samples_seen_ = 0
        self.n_features_in_ = None
        self._XtX = None  # Accumulated X.T @ X matrix

    def compute_mean(self, X_batches):
        """Compute the mean from batches of data (first pass).

        Parameters
        ----------
        X_batches : iterable of torch.Tensor or single torch.Tensor
            Either an iterable yielding batches of data, or a single tensor.
            Each batch should have shape (n_samples, n_features).

        Returns
        -------
        self : ExactIncrementalPCA
            Returns the instance itself.
        """
        total_sum = None
        n_samples_seen = 0

        # Handle single tensor input
        if isinstance(X_batches, (torch.Tensor, np.ndarray)):
            X_batches = [X_batches]

        for batch in X_batches:
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

    def partial_fit(self, X: torch.Tensor):
        """Incrementally fit the model with a batch of samples.

        This method assumes the mean has already been computed using compute_mean().

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
                device=X.device,
            )

        self._XtX += X_centered.T @ X_centered

        return self

    def fit(self, X_batches, y=None):
        """Fit the model with batches of samples.

        This method assumes the mean has already been computed using compute_mean().
        If mean is not computed, it will compute it first (requiring two passes).

        Parameters
        ----------
        X_batches : iterable of torch.Tensor or single torch.Tensor
            Either an iterable yielding batches of data, or a single tensor.
            Each batch should have shape (n_samples, n_features).
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        self : ExactIncrementalPCA
            Returns the instance itself.
        """
        # Handle single tensor input
        if isinstance(X_batches, (torch.Tensor, np.ndarray)):
            X_batches_list = [X_batches]
        else:
            # Convert generator to list to allow multiple passes
            X_batches_list = list(X_batches)

        # Compute mean if not already done
        if self.mean_ is None:
            if self.verbose:
                self.logger.info("Computing mean (first pass through data)")
            self.compute_mean(X_batches_list)

        # Reset XtX for fresh computation
        self._XtX = None

        # Build covariance matrix (second pass)
        if self.verbose:
            self.logger.info("Building covariance matrix (second pass through data)")

        for batch in X_batches_list:
            self.partial_fit(batch)

        # Compute eigendecomposition
        self._compute_components()

        return self

    def _compute_components(self):
        """Compute principal components from the accumulated covariance matrix."""
        if self._XtX is None:
            raise ValueError("No data has been fitted yet")

        # Compute covariance matrix
        covariance = self._XtX / self.n_samples_seen_

        # Compute eigendecomposition
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        # Select top n_components (reverse order for descending eigenvalues)
        n_components = min(self.n_components, self.n_features_in_)
        idx = torch.arange(eigenvalues.shape[0] - 1, -1, -1)[:n_components]

        self.explained_variance_ = eigenvalues[idx]
        self.components_ = eigenvectors[:, idx].T

        # Ensure deterministic output using svd_flip convention
        for i in range(n_components):
            if self.components_[i, 0] < 0:
                self.components_[i] *= -1

        if self.verbose:
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
