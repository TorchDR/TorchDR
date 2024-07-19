# -*- coding: utf-8 -*-
"""Classes for clustering methods."""

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

    @abstractmethod
    def fit(self, X: torch.Tensor | np.ndarray, y=None):
        r"""Fit the clustering model.

        This method must be overridden by subclasses. This base implementation
        only converts the input data :math:`\mathbf{X}` to a torch tensor with
        the right device.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data or input affinity matrix if it is precomputed.
        y : None
            Ignored.

        Returns
        -------
        X_torch : torch.Tensor
            Input data as a torch tensor.
        """
        if self.verbose:
            print(f"[TorchDR] Fitting clustering model {self.__class__.__name__} ...")

        X_torch = to_torch(X, device=self.device)
        return X_torch

    def fit_predict(self, X: torch.Tensor | np.ndarray, y=None):
        """Fit the clustering model and output the predicted labels.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            or (n_samples, n_samples) if precomputed is True
            Input data or input affinity matrix if it is precomputed.
        y : None
            Ignored.

        Returns
        -------
        labels : torch.Tensor or np.ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_


class KMeans(ClusteringModule):
    """Implementation of the k-means algorithm.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to form.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
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
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
        random_state: float = 0,
    ):

        super().__init__(
            n_clusters=n_clusters,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
        )

        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: torch.Tensor | np.ndarray):
        """Fit the k-means model.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : object
            The fitted instance.
        """
        return self

    def _instantiate_generator(self):
        self.generator_ = np.random.default_rng(
            seed=self.random_state
        )  # we use numpy because torch.Generator is not picklable
        return self.generator_
