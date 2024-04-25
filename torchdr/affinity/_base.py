# -*- coding: utf-8 -*-
"""
Common (simple) affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod

from torchdr.utils import pairwise_distances, normalize_matrix


class Affinity(ABC):
    """
    Base class for affinity matrices.
    Each children class should implement a fit method.
    """

    def __init__(self):
        self.log = {}

    @abstractmethod
    def fit(self, X):
        self.data_ = X

    def fit_get(self, X, *args, **kwargs):
        self.fit(X)
        return self.get(X, *args, **kwargs)

    def get(self, X):
        if not hasattr(self, "affinity_matrix_"):
            self.fit(X)
            assert hasattr(
                self, "affinity_matrix_"
            ), "[TorchDR] Affinity (Error) : affinity_matrix_ should be computed "
            "in fit method."
        return self.affinity_matrix_  # type: ignore


class ScalarProductAffinity(Affinity):
    def __init__(self, centering=False, keops=False):
        super().__init__()
        self.centering = centering
        self.keops = keops

    def fit(self, X):
        super().fit(X)
        if self.centering:
            X = X - X.mean(0)
        self.affinity_matrix_ = -pairwise_distances(
            X, metric="angular", keops=self.keops
        )


class LogAffinity(Affinity):
    """Computes an affinity matrix from an affinity matrix in log space."""

    def __init__(self):
        super().__init__()

    def get(self, X, log=False):
        if not hasattr(self, "log_affinity_matrix_"):
            self.fit(X)
            assert hasattr(
                self, "log_affinity_matrix_"
            ), "[TorchDR] Affinity (Error) : log_affinity_matrix_ should be computed \
                in  fit method of a LogAffinity."

        if log:  # return the log of the affinity matrix
            return self.log_affinity_matrix_  # type: ignore
        else:
            if not hasattr(self, "affinity_matrix_"):
                self.affinity_matrix_ = self.log_affinity_matrix_.exp()  # type: ignore
            return self.affinity_matrix_


class GibbsAffinity(LogAffinity):
    def __init__(self, sigma=1.0, metric="euclidean", dim=(0, 1), keops=False):
        super().__init__()
        self.sigma = sigma
        self.metric = metric
        self.dim = dim
        self.keops = keops

    def fit(self, X):
        super().fit(X)
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        log_P = -C / self.sigma
        self.log_affinity_matrix_ = normalize_matrix(log_P, dim=self.dim, log=True)


class StudentAffinity(Affinity):
    def __init__(
        self, degrees_of_freedom=1, metric="euclidean", dim=(0, 1), keops=False
    ):
        super().__init__()
        self.degrees_of_freedom = degrees_of_freedom
        self.metric = metric
        self.dim = dim
        self.keops = keops

    def fit(self, X):
        super().fit(X)
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        C /= self.degrees_of_freedom
        C += 1.0
        C **= -(self.degrees_of_freedom + 1) / 2
        self.affinity_matrix_ = normalize_matrix(C, dim=self.dim, log=False)
