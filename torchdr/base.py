# -*- coding: utf-8 -*-
"""Base classes for DR methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import torch
import numpy as np

from torchdr.utils import pykeops


class DRModule(BaseEstimator, ABC):
    """Base class for DR methods.

    Each children class should implement the fit_transform method.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to project the input data onto.
    device : str, default="auto"
        Device on which the computations are performed.
    keops : bool, default=False
        Whether to use KeOps for computations.
    verbose : bool, default=False
        Whether to print information during the computations.
    random_state : float, default=0
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
        random_state: float = 0,
    ):
        if keops and not pykeops:
            raise ValueError(
                "[TorchDR] ERROR : pykeops is not installed. Please install it to use "
                "`keops=true`."
            )

        self.n_components = n_components
        self.device = device
        self.keops = keops
        self.random_state = random_state
        self.verbose = verbose
        if self.verbose:
            print(f"[TorchDR] Initializing DR model {self.__class__.__name__} ")

    @abstractmethod
    def fit_transform(self, X: torch.Tensor | np.ndarray, y=None):
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
