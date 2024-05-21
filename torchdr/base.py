# -*- coding: utf-8 -*-
"""
Base classes for DR methods
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

from torchdr.utils import to_torch


class DRModule(TransformerMixin, BaseEstimator, ABC):
    """
    Base class for DR methods.
    Each children class should implement the fit method.
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        self.n_components = n_components
        self.device = device
        self.keops = keops
        self.verbose = verbose

    def _process_input(self, X):
        self.data_, self.input_backend_, self.input_device_ = to_torch(
            X, device=self.device, verbose=self.verbose, return_backend_device=True
        )
        self.n_features_ = self.data_.shape[1]
        return self

    @abstractmethod
    def fit(self, X, y=None):
        """Projects input data X onto a low-dimensional space.

        Parameters
        ----------
        X : array-like object of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
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
        return self
