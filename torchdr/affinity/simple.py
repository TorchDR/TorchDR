# -*- coding: utf-8 -*-
"""
Common simple affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from typing import Tuple

from torchdr.utils import logsumexp_red
from torchdr.affinity.base import Affinity, LogAffinity
from torchdr.utils import extract_batch_normalization


def _log_Gibbs(C, sigma):
    r"""
    Returns the Gibbs affinity matrix in log domain.
    """
    return -C / sigma


def _log_Student(C, degrees_of_freedom):
    r"""
    Returns the Student affinity matrix in log domain.
    """
    return -0.5 * (degrees_of_freedom + 1) * (C / degrees_of_freedom + 1).log()


class ScalarProductAffinity(Affinity):
    r"""
    Computes the scalar product affinity matrix :math:`\mathbf{X} \mathbf{X}^T`
    where :math:`\mathbf{X}` is the input data.
    """

    def __init__(
        self,
        device: str = "cuda",
        keops: bool = True,
        verbose: bool = True,
        centering: bool = False,
    ):
        super().__init__(metric="angular", device=device, keops=keops, verbose=verbose)
        self.centering = centering

    def fit(self, X: torch.Tensor | np.ndarray):
        super().fit(X)
        if self.centering:
            self.data_ = self.data_ - self.data_.mean(0)
        self.affinity_matrix_ = -self._pairwise_distance_matrix(self.data_)


class GibbsAffinity(LogAffinity):
    r"""
    Computes the Gibbs affinity matrix :math:`\exp( - \mathbf{C} / \sigma)`
    where :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma` is the bandwidth parameter, normalized according to the
    specified normalization dimensions.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth parameter.
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix.
    metric : str, optional
        Metric to use for pairwise distances computation.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        normalization_dim: int | Tuple[int] = (0, 1),
        metric: str = "euclidean",
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.sigma = sigma
        self.normalization_dim = normalization_dim

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the Gibbs affinity model to the provided data.

        This method computes the pairwise distance matrix :math:`\mathbf{C}`
        for the input data, and then calculates the Gibbs affinity matrix
        :math:`\exp( - \mathbf{C} / \sigma)`.
        The affinity matrix is then normalized according to the specified
        normalization dimensions.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data, either as a PyTorch tensor or a NumPy array. The shape of `X` should be
            (n_samples, n_features).

        Returns
        -------
        self : GibbsAffinity
            The fitted Gibbs affinity model.
        """
        super().fit(X)
        C = self._pairwise_distance_matrix(self.data_)
        log_P = _log_Gibbs(C, self.sigma)
        self.log_normalization_ = logsumexp_red(log_P, self.normalization_dim)
        self.log_affinity_matrix_ = log_P - self.log_normalization_
        return self

    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Returns the batched version of the fitted Gibbs affinity matrix.

        Parameters
        ----------
        indices : torch.Tensor of shape (n_batch, batch_size)
            Indices of the batch.
        log : bool, optional
            If True, returns the log of the affinity matrix.

        Returns
        -------
        P_batch : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_batch, batch_size, batch_size)
            The affinity matrix for the batch indices.
            In log domain if `log` is True.
        """
        C_batch = super().get_batch(indices)
        log_P_batch = _log_Gibbs(C_batch, self.sigma)
        log_normalization_batch = extract_batch_normalization(
            self.log_normalization_, indices, self.normalization_dim
        )
        log_P_batch -= log_normalization_batch

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()


class StudentAffinity(LogAffinity):
    r"""
    Computes the Student affinity matrix based on the Student-t distribution.

    Parameters
    ----------
    degrees_of_freedom : int, optional
        Degrees of freedom for the Student-t distribution.
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix.
    metric : str, optional
        Metric to use for pairwise distances computation.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity.
    """

    def __init__(
        self,
        degrees_of_freedom: int = 1,
        normalization_dim: int | Tuple[int] = (0, 1),
        metric: str = "euclidean",
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.normalization_dim = normalization_dim
        self.degrees_of_freedom = degrees_of_freedom

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the Student affinity model to the provided data.

        This method computes the pairwise distance matrix :math:`\mathbf{C}`
        for the input data, then calculates the Student affinity matrix based
        on the Student-t distribution:
        :math:`\left(1 + \frac{\mathbf{C}}{\text{degrees_of_freedom}}\right)^{-\frac{\text{degrees_of_freedom} + 1}{2}}`.
        The affinity matrix is then normalized according to the specified
        normalization dimensions.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data, either as a PyTorch tensor or a NumPy array. The shape of `X` should be
            (n_samples, n_features).

        Returns
        -------
        self : StudentAffinity
            The fitted Student affinity model.
        """
        super().fit(X)
        C = self._pairwise_distance_matrix(self.data_)
        log_P = _log_Student(C, self.degrees_of_freedom)
        self.log_normalization_ = logsumexp_red(log_P, self.normalization_dim)
        self.log_affinity_matrix_ = log_P - self.log_normalization_
        return self

    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Returns the batched version of the fitted Student affinity matrix.

        Parameters
        ----------
        indices : torch.Tensor of shape (n_batch, batch_size)
            Indices of the batch.
        log : bool, optional
            If True, returns the log of the affinity matrix.

        Returns
        -------
        P_batch : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_batch, batch_size, batch_size)
            The affinity matrix for the batch indices.
            In log domain if `log` is True.
        """
        C_batch = super().get_batch(indices)
        log_P_batch = _log_Student(C_batch, self.degrees_of_freedom)
        log_normalization_batch = extract_batch_normalization(
            self.log_normalization_, indices, self.normalization_dim
        )
        log_P_batch -= log_normalization_batch

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()
