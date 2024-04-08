# -*- coding: utf-8 -*-
"""
Common (simple) affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from abc import ABC, abstractmethod


class BaseAffinity(ABC):
    def __init__(self):
        self.log_ = {}

    # @abstractmethod
    # def fit(self, X):
    #     pass

    # def fit_transform(self, X):
    #     self.fit(X)
    #     return self.P

    @abstractmethod
    def compute_affinity(self, X):
        pass


class GramAffinity(BaseAffinity):
    def __init__(self, centering=False):
        super(GramAffinity, self).__init__()
        self.centering = centering

    def compute_affinity(self, X):
        if self.centering:
            X = X - X.mean(0)
        self.affinity_matrix = X @ X.T
        return self.affinity_matrix


class LogAffinity(BaseAffinity):
    @abstractmethod
    def compute_log_affinity(self, X):
        pass

    def compute_affinity(self, X):
        """Computes an affinity matrix from an affinity matrix in log space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        P : torch.Tensor of shape (n_samples, n_samples)
            Affinity matrix.
        """
        log_P = self.compute_log_affinity(X)
        self.log_affinity_matrix = log_P
        if log_P is not None:
            self.affinity_matrix = log_P.exp()
            return self.affinity_matrix
        else:
            return None


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
