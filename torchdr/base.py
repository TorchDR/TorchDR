# -*- coding: utf-8 -*-
"""
Base classes for DR methods
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

from torchdr.utils import to_torch


class DRModule(ABC, BaseEstimator):
    """
    Base class for DR methods.
    Each children class should implement the fit and transform methods.
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        self.log = {}
        self.n_components = n_components
        self.device = device
        self.keops = keops
        self.verbose = verbose

    @abstractmethod
    def fit(self, X, y=None):
        """Projects input data X onto a low-dimensional space.

        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            or tensor of shape (n_samples, n_samples)
            Input data or input affinity matrix if it is precomputed.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        if self.verbose:
            print("[TorchDR] Fitting DR model ...")
        self.data_, self.input_backend, self.input_device = to_torch(
            X, device=self.device, verbose=self.verbose, return_backend_device=True
        )
        return self

    @abstractmethod
    def transform(self, X):
        pass

    def forward(self, X):
        return self.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
