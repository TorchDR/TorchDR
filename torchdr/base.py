"""Base class for dimensionality reduction methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader

from torchdr.utils import (
    seed_everything,
    set_logger,
    handle_input_output,
)
from torchdr.distance import FaissConfig

from typing import Optional, Any, TypeVar, Union

ArrayLike = TypeVar("ArrayLike", torch.Tensor, np.ndarray)


class DRModule(BaseEstimator, nn.Module, ABC):
    """Base class for dimensionality reduction methods.

    Subclasses must implement :meth:`_fit_transform`.

    Parameters
    ----------
    n_components : int, optional
        Number of dimensions for the embedding. Default is 2.
    device : str, optional
        Device for computations. ``"auto"`` uses the input tensor's device.
        Default is "auto".
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Backend for handling sparsity and memory efficiency.
        Default is None (standard PyTorch).
    verbose : bool, optional
        Verbosity. Default is False.
    random_state : float, optional
        Random seed for reproducibility. Default is None.
    compile : bool, default=False
        Whether to use ``torch.compile`` for faster computation.
    process_duplicates : bool, default=True
        Whether to handle duplicate data points by default.
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        backend: Union[str, FaissConfig, None] = None,
        verbose: bool = False,
        random_state: Optional[float] = None,
        compile: bool = False,
        process_duplicates: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.n_components = n_components
        self.device = device if device is not None else "auto"
        self.backend = backend
        self.verbose = verbose
        self.random_state = random_state
        self.compile = compile
        self.process_duplicates = process_duplicates

        self.logger = set_logger(self.__class__.__name__, self.verbose)

        if self.random_state is not None:
            self._actual_seed = seed_everything(
                self.random_state, fast=True, deterministic=False
            )
            self.logger.info(f"Random seed set to: {self._actual_seed}.")

        self.embedding_ = None
        self.is_fitted_ = False

    # --- Public API ---

    @handle_input_output()
    def fit(self, X: ArrayLike, y: Optional[Any] = None) -> "DRModule":
        """Fit the model from the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data (or ``(n_samples, n_samples)`` if precomputed).
        y : None
            Ignored.

        Returns
        -------
        self : DRModule
            The fitted instance.
        """
        self.fit_transform(X, y=y)
        return self

    @handle_input_output()
    def fit_transform(self, X: ArrayLike, y: Optional[Any] = None) -> ArrayLike:
        """Fit the model and return the embedding.

        Handles duplicate data points by default: performs DR on unique
        points and maps results back to the original structure. Controlled
        by :attr:`process_duplicates`.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data (or ``(n_samples, n_samples)`` if precomputed).
        y : None
            Ignored.

        Returns
        -------
        embedding_ : ArrayLike of shape (n_samples, n_components)
            The embedding.
        """
        if self.process_duplicates and isinstance(X, DataLoader):
            self.logger.warning(
                "process_duplicates is not supported with DataLoader input. "
                "Consider deduplicating your dataset before creating "
                "the DataLoader."
            )

        if self.process_duplicates and not isinstance(X, DataLoader):
            X_unique, inverse_indices = torch.unique(X, dim=0, return_inverse=True)
            if X_unique.shape[0] < X.shape[0]:
                n_duplicates = X.shape[0] - X_unique.shape[0]
                self.logger.info(
                    f"Detected {n_duplicates} duplicate samples, "
                    "performing DR on unique data."
                )
                embedding_unique = self._fit_transform(X_unique, y=y)
                if isinstance(self.embedding_, torch.nn.Parameter):
                    self.embedding_.data = embedding_unique[inverse_indices]
                else:
                    self.embedding_ = embedding_unique[inverse_indices]
            else:
                self.embedding_ = self._fit_transform(X, y=y)
        else:
            self.embedding_ = self._fit_transform(X, y=y)

        self.is_fitted_ = True
        return self.embedding_

    def transform(self, X: Optional[ArrayLike] = None) -> ArrayLike:
        """Transform data into the learned embedding space.

        If ``X`` is None, returns the training embedding. When an encoder
        is set, new data is transformed via ``encoder(X)``.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features), optional
            Data to transform. If None, returns the training embedding.

        Returns
        -------
        embedding_ : ArrayLike of shape (n_samples, n_components)
            The embedding.
        """
        if not self.is_fitted_:
            raise ValueError(
                "This DRModule instance is not fitted yet. "
                "Call 'fit' or 'fit_transform' with some data first."
            )

        if X is not None:
            if getattr(self, "encoder", None) is not None:
                from torchdr.utils import to_torch

                X_tensor = to_torch(X).to(device=self.device_)
                with torch.no_grad():
                    return self.encoder(X_tensor)
            raise NotImplementedError(
                "Transforming new data is not implemented for this model."
            )

        return self.embedding_

    # --- Core algorithm (must be implemented by subclasses) ---

    @abstractmethod
    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None) -> torch.Tensor:
        """Fit the model and return the embedding (core algorithm).

        Subclasses implement this with the actual DR logic. Called by
        :meth:`fit_transform` after duplicate handling.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data (or ``(n_samples, n_samples)`` if precomputed).
        y : None
            Ignored.

        Returns
        -------
        embedding_ : torch.Tensor of shape (n_samples, n_components)
            The embedding.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : _fit_transform method is not implemented."
        )

    # --- Utilities ---

    def _get_compute_device(self, X: torch.Tensor):
        """Return the target device (from ``self.device`` or inferred from X)."""
        return X.device if self.device == "auto" else self.device

    # --- Memory management ---

    def clear_memory(self):
        """Clear non-persistent buffers to free memory after training."""
        if hasattr(self, "_non_persistent_buffers_set"):
            for name in list(self._non_persistent_buffers_set):
                if hasattr(self, name):
                    delattr(self, name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
