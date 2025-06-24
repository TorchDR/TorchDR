"""Base classes for DR methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.base import BaseEstimator

from torchdr.utils import bool_arg, seed_everything

from typing import Union, Optional, Any


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
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        random_state: float = None,
    ):
        self.n_components = n_components
        self.device = device
        self.backend = backend
        self.random_state = random_state

        self.verbose = bool_arg(verbose)
        if self.verbose:
            print(f"[TorchDR] Initializing DR model {self.__class__.__name__}. ")

        if random_state is not None:
            self._actual_seed = seed_everything(
                random_state, fast=True, deterministic=False
            )
            if self.verbose:
                print(f"[TorchDR] Random seed set to: {self._actual_seed}.")

    @abstractmethod
    def fit_transform(
        self, X: Union[torch.Tensor, np.ndarray], y: Optional[Any] = None
    ):
        """Fit the dimensionality reduction model and transform the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data or input affinity matrix if it is precomputed.
        y : None
            Ignored.

        Raises
        ------
        NotImplementedError
            This method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : fit_transform method is not implemented."
        )

    @property
    def embedding_(self) -> torch.Tensor:
        """Embedding of the data.

        Returns
        -------
        torch.Tensor
            Embedding of the data.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : embedding_ property is not implemented."
        )
