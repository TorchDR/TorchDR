# -*- coding: utf-8 -*-
"""k-means algorithm."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

from torchdr.utils import to_torch, pykeops
from torchdr.clustering.base import ClusteringModule


class KMeans(ClusteringModule):
    """Implementation of the k-means algorithm."""

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
