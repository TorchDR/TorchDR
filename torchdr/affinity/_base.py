# -*- coding: utf-8 -*-
"""
Common (simple) affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod

from torchdr.utils import pairwise_distances


class BaseAffinity(ABC):
    def __init__(self):
        self.log = {}

    @abstractmethod
    def fit(self, X):
        self.data_ = X

    def get(self, X):
        if not hasattr(self, "affinity_matrix_"):
            self.fit(X)
            assert hasattr(
                self, "affinity_matrix_"
            ), "affinity_matrix_ should be computed in fit method"
        return self.affinity_matrix_  # type: ignore


class ScalarProductAffinity(BaseAffinity):
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


class LogAffinity(BaseAffinity):
    """Computes an affinity matrix from an affinity matrix in log space."""

    def __init__(self):
        super().__init__()

    def get(self, X, log=False):
        if not hasattr(self, "log_affinity_matrix_"):
            self.fit(X)
            assert hasattr(
                self, "log_affinity_matrix_"
            ), "log_affinity_matrix_ should be computed in fit method of a LogAffinity"

        if log:  # return the log of the affinity matrix
            return self.log_affinity_matrix_  # type: ignore
        else:
            if not hasattr(self, "affinity_matrix_"):
                self.affinity_matrix_ = self.log_affinity_matrix_.exp()  # type: ignore
            return self.affinity_matrix_


class GibbsAffinity(LogAffinity):
    def __init__(self, sigma=1.0, metric="euclidean", keops=False):
        super().__init__()
        self.sigma = sigma
        self.metric = metric
        self.keops = keops

    def fit(self, X):
        super().fit(X)
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        log_P = -C / self.sigma
        self.log_affinity_matrix_ = log_P - log_P.logsumexp(1)[:, None]


class StudentAffinity(LogAffinity):
    def __init__(self, metric="euclidean", keops=False):
        super().__init__()
        self.metric = metric
        self.keops = keops

    def fit(self, X):
        super().fit(X)
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        log_P = -(1 + C).log()
        self.log_affinity_matrix_ = log_P - log_P.logsumexp(1)[:, None]
