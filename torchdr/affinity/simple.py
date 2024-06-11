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

from torchdr.utils import logsumexp_red, sum_red
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
    where :math:`\mathbf{X}` is the input data, normalized according to the
    specified normalization dimensions.

    Parameters
    ----------
    device : str, optional
        Device to use for computations. Default is "cuda".
    keops : bool, optional
        Whether to use KeOps for computations. Default is True.
    verbose : bool, optional
        Verbosity. Default is True.
    centering : bool, optional
        Whether to center the data by subtracting the mean. Default is False.
    """

    def __init__(
        self,
        normalization_dim: int | Tuple[int] = None,
        device: str = "cuda",
        keops: bool = True,
        verbose: bool = True,
        centering: bool = False,
    ):
        super().__init__(metric="angular", device=device, keops=keops, verbose=verbose)
        self.normalization_dim = normalization_dim
        self.centering = centering

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the scalar product affinity model to the provided data.

        This method computes the scalar product affinity matrix
        :math:`\mathbf{X} \mathbf{X}^T` for the input data. If centering is
        enabled, the data is centered by subtracting the mean before computing
        the affinity matrix.
        The affinity matrix is then normalized according to the specified
        normalization dimensions.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data, either as a PyTorch tensor or a NumPy array. The shape
            of `X` should be (n_samples, n_features).

        Returns
        -------
        self : ScalarProductAffinity
            The fitted scalar product affinity model.
        """
        super().fit(X)
        if self.centering:
            self.data_ = self.data_ - self.data_.mean(0)
        self.affinity_matrix_ = -self._pairwise_distance_matrix(self.data_)

        if self.normalization_dim is not None:
            self.normalization_ = sum_red(P, self.normalization_dim)
            self.affinity_matrix_ /= self.normalization_

        return self

    def get_batch(self, indices: torch.Tensor):
        r"""
        Returns the batched version of the fitted scalar product affinity matrix.

        Parameters
        ----------
        indices : torch.Tensor of shape (n_batch, batch_size)
            Indices of the batch.

        Returns
        -------
        P_batch : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_batch, batch_size, batch_size)
            The affinity matrix for the batch indices.
        """
        C_batch = super().get_batch(indices)
        P_batch = -C_batch

        if self.normalization_dim is not None:
            normalization_batch = extract_batch_normalization(
                self.normalization_, indices, self.normalization_dim
            )
            P_batch /= normalization_batch

        return P_batch


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
        self.log_affinity_matrix_ = _log_Gibbs(C, self.sigma)

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                self.log_affinity_matrix_, self.normalization_dim
            )
            self.log_affinity_matrix_ -= self.log_normalization_

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

        if self.normalization_dim is not None:
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
        self.log_affinity_matrix_ = _log_Student(C, self.degrees_of_freedom)

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                self.log_affinity_matrix_, self.normalization_dim
            )
            self.log_affinity_matrix_ -= self.log_normalization_

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

        if self.normalization_dim is not None:
            log_normalization_batch = extract_batch_normalization(
                self.log_normalization_, indices, self.normalization_dim
            )
            log_P_batch -= log_normalization_batch

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()
