"""Base classes for DR methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod

import torch
import numpy as np
from sklearn.base import BaseEstimator

from torchdr.utils import bool_arg, seed_everything, set_logger, handle_type

from typing import Optional, Any, TypeVar

ArrayLike = TypeVar("ArrayLike", torch.Tensor, np.ndarray)


class DRModule(BaseEstimator, ABC):
    """Base class for DR methods.

    Each children class should implement the fit_transform method.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to project the input data onto.
    device : str, default="auto"
        Device on which the computations are performed.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, default=False
        Whether to print information during the computations.
    random_state : float, default=None
        Random seed for reproducibility.
    process_duplicates : bool, default=True
        Whether to handle duplicate data points by default.
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        random_state: float = None,
        process_duplicates: bool = True,
    ):
        self.n_components = n_components
        self.device = device
        self.backend = backend
        self.random_state = random_state
        self.verbose = bool_arg(verbose)
        self.process_duplicates = process_duplicates

        self.logger = set_logger(self.__class__.__name__, self.verbose)

        if random_state is not None:
            self._actual_seed = seed_everything(
                random_state, fast=True, deterministic=False
            )
            self.logger.info(f"Random seed set to: {self._actual_seed}.")

        self.embedding_ = None
        self.is_fitted_ = False

    @abstractmethod
    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        """Fit the dimensionality reduction model and transform the input data.

        This method should be implemented by subclasses and contains the core
        logic for the DR algorithm, assuming unique data points.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data or input affinity matrix if it is precomputed.
        y : None
            Ignored.

        Returns
        -------
        embedding_ : torch.Tensor of shape (n_samples, n_components)
            The embedding of the input data in the lower-dimensional space.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : _fit_transform method is not implemented."
        )

    @handle_type(
        accept_sparse=False,
        ensure_min_samples=2,
        ensure_min_features=1,
        ensure_2d=True,
    )
    def fit(self, X: ArrayLike, y: Optional[Any] = None) -> "DRModule":
        """Fit the dimensionality reduction model from the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data or input affinity matrix if it is precomputed.
        y : None
            Ignored.

        Returns
        -------
        self : DRModule
            The fitted DRModule instance.
        """
        self.fit_transform(X, y=y)
        return self

    @handle_type(
        accept_sparse=False,
        ensure_min_samples=2,
        ensure_min_features=1,
        ensure_2d=True,
    )
    def fit_transform(self, X: ArrayLike, y: Optional[Any] = None) -> ArrayLike:
        """Fit the dimensionality reduction model and transform the input data.

        This method handles duplicate data points by default. It performs
        dimensionality reduction on unique data points and then maps the
        results back to the original data structure. This behavior can be
        controlled by the `process_duplicates` parameter.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data or input affinity matrix if it is precomputed.
        y : None
            Ignored.

        Returns
        -------
        embedding_ : ArrayLike of shape (n_samples, n_components)
            The embedding of the input data in the lower-dimensional space.
        """
        if self.process_duplicates:
            X_unique, inverse_indices = torch.unique(X, dim=0, return_inverse=True)
            if X_unique.shape[0] < X.shape[0]:
                n_duplicates = X.shape[0] - X_unique.shape[0]
                self.logger.info(
                    f"Detected {n_duplicates} duplicate samples, "
                    "performing DR on unique data."
                )
                embedding_unique = self._fit_transform(X_unique, y=y)
                self.embedding_ = embedding_unique[inverse_indices]
            else:
                self.embedding_ = self._fit_transform(X, y=y)
        else:
            self.embedding_ = self._fit_transform(X, y=y)

        self.is_fitted_ = True
        return self.embedding_

    def transform(self, X: Optional[ArrayLike] = None) -> ArrayLike:
        """Transform the input data into the learned embedding space.

        This method can only be called after the model has been fitted.
        If `X` is not provided, it returns the embedding of the training data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features), optional
            The data to transform. If None, returns the training data embedding.
            Not all models support transforming new data.

        Returns
        -------
        embedding_ : ArrayLike of shape (n_samples, n_components)
            The embedding of the input data.

        Raises
        ------
        NotImplementedError
            If the model does not support transforming new data.
        ValueError
            If the model has not been fitted yet.
        """
        if not self.is_fitted_:
            raise ValueError(
                "This DRModule instance is not fitted yet. "
                "Call 'fit' or 'fit_transform' with some data first."
            )

        if X is not None:
            raise NotImplementedError(
                "Transforming new data is not implemented for this model."
            )

        return self.embedding_
