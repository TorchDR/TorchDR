# -*- coding: utf-8 -*-
"""Base classes for clustering methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

from torchdr.utils import to_torch, pykeops


class ClusteringModule(BaseEstimator, ABC):
    """Base class for clustering methods.

    Each children class should implement the fit method.

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
        n_clusters: int = 2,
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

        self.n_clusters = n_clusters
        self.device = device
        self.keops = keops
        self.verbose = verbose
        self.random_state = random_state

    def fit_predict(self, X, y=None, **kwargs):
        """Perform clustering on input data and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by convention.

        **kwargs : dict
            Arguments to be passed to ``fit``.

            .. versionadded:: 1.4

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=np.int64
            Cluster labels.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X, **kwargs)
        return self.labels_
