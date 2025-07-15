"""Classes for clustering methods."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.base import BaseEstimator

from torchdr.utils import (
    kmin,
    pykeops,
    faiss,
    seed_everything,
    set_logger,
    bool_arg,
    handle_type,
)
from torchdr.distance import pairwise_distances

from typing import Optional, Any, TypeVar

ArrayLike = TypeVar("ArrayLike", torch.Tensor, np.ndarray)


class ClusteringModule(BaseEstimator, ABC):
    """Base class for clustering methods.

    Each child class should implement the fit method.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to form.
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
        n_clusters: int = 2,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        random_state: float = None,
    ):
        if backend == "keops" and not pykeops:
            raise ValueError(
                "[TorchDR] ERROR: pykeops is not installed. Please install it to use "
                "`keops=True`."
            )

        if backend == "faiss" and not faiss:
            raise ValueError(
                "[TorchDR] ERROR: faiss is not installed. Please install it to use "
                "`backend=faiss`."
            )

        self.n_clusters = n_clusters
        self.device = device
        self.backend = backend
        self.random_state = random_state
        self.verbose = bool_arg(verbose)

        # --- Logger setup ---
        self.logger = set_logger(self.__class__.__name__, self.verbose)

        if random_state is not None:
            self._actual_seed = seed_everything(
                random_state, fast=True, deterministic=False
            )
            self.logger.info(f"Random seed set to: {self._actual_seed}.")

    @abstractmethod
    def _fit(self, X: torch.Tensor, y: Optional[Any] = None):
        """Fit the clustering model.

        This method should be implemented by subclasses and contains the core
        logic for the clustering algorithm.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            The fitted instance.
        """
        raise NotImplementedError

    @handle_type()
    def fit(self, X: ArrayLike, y: Optional[Any] = None):
        """Fit the clustering model.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            The fitted instance.
        """
        return self._fit(X, y)

    @handle_type()
    def fit_predict(self, X: ArrayLike, y: Optional[Any] = None):
        """Fit the clustering model and output the predicted labels.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.

        Returns
        -------
        labels : ArrayLike of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_


class KMeans(ClusteringModule):
    """Implementation of the k-means algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form.
    init : {'random', 'k-means++'}, default='k-means++'
        Method for initialization.
        - 'random': choose `n_clusters` observations (rows) at random from data
          for the initial centroids.
        - 'k-means++': selects initial cluster centers for k-means clustering
          in a smart way to speed up convergence.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    device : str, default="auto"
        Device on which the computations are performed.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, default=False
        Whether to print information during the computations.
    random_state : float, default=None
        Random seed for reproducibility.
    metric : str, default="sqeuclidean"
        Metric to use for the distance computation.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init="k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        random_state: float = None,
        metric: str = "sqeuclidean",
    ):
        super().__init__(
            n_clusters=n_clusters,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
        )

        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        if metric not in ["sqeuclidean", "euclidean"]:
            raise ValueError(
                f"[TorchDR] Metric '{metric}' not supported for KMeans. "
                "Expected 'sqeuclidean' or 'euclidean'."
            )
        self.metric = metric

    def _fit(self, X: torch.Tensor, y: Optional[Any] = None):
        """Fit the k-means model.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            The input data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            The fitted instance.
        """
        self.inertia_ = float("inf")

        for _ in range(self.n_init):
            centroids, centroid_membership = self._fit_single(X)
            inertia = self._compute_inertia(X, centroids, centroid_membership)

            if inertia < self.inertia_:
                self.inertia_ = inertia
                self.labels_ = centroid_membership
                self.cluster_centers_ = centroids

        return self

    def _compute_inertia(self, X, centroids, centroid_membership):
        assigned_centroids = centroids[centroid_membership]
        # the following works with pykeops LazyTensors
        if self.metric == "sqeuclidean":
            distances = (X - assigned_centroids).pow(2).sum(dim=1)
        elif self.metric == "euclidean":
            distances = (X - assigned_centroids).pow(2).sum(dim=1).sqrt()
        inertia = distances.sum()
        return inertia

    def _fit_single(self, X):
        n_samples_in, n_features_in = X.shape

        centroids = self._init_centroids(X)

        for it in range(self.max_iter):
            # E step: assign points to the closest cluster
            C = pairwise_distances(
                X, centroids, metric=self.metric, backend=self.backend
            )
            _, centroid_membership = kmin(C, k=1, dim=1)
            centroid_membership = centroid_membership.view(-1).to(torch.int64)

            # M step: update the centroids to the normalized cluster average
            # Compute the sum of points per cluster:
            new_centroids = torch.zeros_like(centroids)
            new_centroids.scatter_add_(
                0, centroid_membership[:, None].repeat(1, n_features_in), X
            )
            # Divide by the number of points per cluster:
            Ncl = (
                torch.bincount(centroid_membership, minlength=self.n_clusters)
                .type_as(new_centroids)
                .view(self.n_clusters, 1)
            )
            new_centroids /= Ncl

            if torch.allclose(new_centroids, centroids, atol=self.tol):
                break

            centroids = new_centroids

        return centroids, centroid_membership

    def _init_centroids(self, X):
        n_samples, n_features = X.shape

        if self.init == "random":
            centroid_indices = torch.randperm(n_samples, device=X.device)[
                : self.n_clusters
            ]
            centroids = X[centroid_indices].clone()
        elif self.init == "k-means++":
            centroids = self._kmeans_plusplus(X)
        else:
            raise ValueError(
                f"Unknown init method '{self.init}'. Expected 'random' or 'k-means++'."
            )
        return centroids

    def _kmeans_plusplus(self, X):
        n_samples, n_features = X.shape
        centers = torch.empty(
            (self.n_clusters, n_features), device=X.device, dtype=X.dtype
        )

        # Randomly choose the first centroid
        center_id = torch.randint(0, n_samples, (1,), device=X.device).item()
        centers[0] = X[center_id]

        # Initialize list of closest distances
        closest_dist_sq = pairwise_distances(
            X, centers[0:1], metric=self.metric, backend=None
        ).squeeze()

        for c in range(1, self.n_clusters):
            # Choose the next centroid
            # Compute probabilities proportional to squared distances
            probs = closest_dist_sq / torch.sum(closest_dist_sq)
            probs = torch.clamp(probs, min=0)

            center_id = torch.multinomial(probs, 1).item()
            centers[c] = X[center_id]

            # Update the closest distances
            distances = pairwise_distances(
                X, centers[c : c + 1], metric=self.metric, backend=None
            ).squeeze()

            if self.metric == "euclidean":
                distances = distances**2
            elif self.metric == "sqeuclidean":
                pass  # distances are already squared

            closest_dist_sq = torch.minimum(closest_dist_sq, distances)

        return centers

    @handle_type()
    def predict(self, X: ArrayLike):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        labels : ArrayLike of shape (n_samples,)
            Cluster labels.
        """
        C = pairwise_distances(
            X, self.cluster_centers_, metric=self.metric, backend=None
        )
        _, labels = kmin(C, k=1, dim=1)
        return labels.view(-1).to(torch.int64)
