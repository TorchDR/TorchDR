# -*- coding: utf-8 -*-
"""
Common (simple) affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from abc import ABC, abstractmethod

from torchdr.utils.geometry import pairwise_distances


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
    def __init__(self, centering=False):
        super(ScalarProductAffinity, self).__init__()
        self.centering = centering

    def fit(self, X):
        super(ScalarProductAffinity, self).fit(X)
        if self.centering:
            X = X - X.mean(0)
        self.affinity_matrix_ = X @ X.T


class LogAffinity(BaseAffinity):
    """Computes an affinity matrix from an affinity matrix in log space."""

    def __init__(self):
        super(LogAffinity, self).__init__()

    def get(self, X, log=True):
        if not hasattr(self, "log_affinity_matrix_"):
            self.fit(X)
            assert hasattr(
                self, "log_affinity_matrix_"
            ), "log_affinity_matrix_ should be computed in fit method of a LogAffinity"

        if log:
            return self.log_affinity_matrix_  # type: ignore
        else:
            if not hasattr(self, "affinity_matrix_"):
                self.affinity_matrix_ = self.log_affinity_matrix_.exp()  # type: ignore
            return self.affinity_matrix_


class GibbsAffinity(LogAffinity):
    def __init__(self, sigma=1.0, dim=(0, 1), metric="euclidean", keops=False):
        super(GibbsAffinity, self).__init__()
        self.sigma = sigma
        self.dim = dim
        self.metric = metric
        self.keops = keops

    def fit(self, X):
        super(GibbsAffinity, self).fit(X)
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        log_P = -C / self.sigma
        self.log_affinity_matrix_ = log_P - log_P.logsumexp(dim=self.dim)


class StudentAffinity(LogAffinity):
    def __init__(self, metric="euclidean", keops=False):
        super(StudentAffinity, self).__init__()
        self.dim = (0, 1)
        self.metric = metric
        self.keops = keops

    def fit(self, X):
        super(StudentAffinity, self).fit(X)
        C = pairwise_distances(X, metric=self.metric, keops=self.keops)
        log_P = -(1 + C).log()
        self.log_affinity_matrix_ = log_P - log_P.logsumexp(dim=self.dim)


class NormalizedGaussianAndStudentAffinity(LogAffinity):
    """
    This class computes the normalized affinity associated to a Gaussian or t-Student
    kernel. The affinity matrix is normalized by given axis.

    Parameters
    ----------
    student : bool, optional
        If True, computes a t-Student kernel (default False).
    sigma : float, optional
        The length scale of the Gaussian kernel (default 1.0).
    p : int, optional
        p value for the p-norm distance to calculate between each vector pair
        (default 2).
    """

    def __init__(self, student=False, sigma=1.0, p=2):
        self.student = student
        self.sigma = sigma
        self.p = p
        super(NormalizedGaussianAndStudentAffinity, self).__init__()

    def compute_log_affinity(self, X, axis=(0, 1)):
        """
        Computes the pairwise affinity matrix in log space and normalize it by given
        axis.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        log_P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix in log space.
        """
        C = torch.cdist(X, X, self.p) ** 2
        if self.student:
            log_P = -torch.log(1 + C)
        else:
            log_P = -C / (2 * self.sigma)
        return log_P - torch.logsumexp(log_P, dim=axis)
