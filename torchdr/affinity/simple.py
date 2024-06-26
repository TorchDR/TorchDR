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
from torchdr.utils import (
    extract_batch_normalization,
    kmin,
    wrap_vectors,
    batch_transpose,
)


def _log_Gibbs(C, sigma):
    r"""
    Returns the Gibbs affinity matrix in log domain.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        Pairwise distance matrix.
    sigma : float
        Bandwidth parameter.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor
        The Gibbs affinity matrix in log domain.
    """
    return -C / sigma


@wrap_vectors
def _log_LocalGibbs(C, sigma):
    r"""
    Returns the Local Gibbs affinity matrix with sample-wise bandwidth
    determined by the distance from a point to its K-th neirest neighbor
    in log domain.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        Pairwise distance matrix.
    sigma : torch.Tensor of shape (n,)
        Sample-wise bandwidth parameter.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor
    """
    sigma_t = batch_transpose(sigma)
    return -C / (sigma * sigma_t)


def _log_Student(C, degrees_of_freedom):
    r"""
    Returns the Student affinity matrix in log domain.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        Pairwise distance matrix.
    degrees_of_freedom : float
        Degrees of freedom parameter.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor
    """
    return -0.5 * (degrees_of_freedom + 1) * (C / degrees_of_freedom + 1).log()


class ScalarProductAffinity(Affinity):
    r"""
    Computes the scalar product affinity matrix :math:`\mathbf{X} \mathbf{X}^\top`
    where :math:`\mathbf{X}` is the input data. The affinity can be normalized
    according to the specified normalization dimensions.

    Parameters
    ----------
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix. Default is None.
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
        super().__init__(
            metric="angular", device=device, keops=keops, verbose=verbose, nodiag=False
        )
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
            Input data.

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
            self.normalization_ = sum_red(self.affinity_matrix_, self.normalization_dim)
            self.affinity_matrix_ = self.affinity_matrix_ / self.normalization_

        return self

    def get_batch(self, indices: torch.Tensor):
        r"""
        Extracts the affinity submatrix corresponding to the indices.

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
            P_batch = P_batch / normalization_batch

        return P_batch


class GibbsAffinity(LogAffinity):
    r"""
    Computes the Gibbs affinity matrix :math:`\exp( - \mathbf{C} / \sigma)`
    where :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma` is the bandwidth parameter. The affinity can be normalized
    according to the specified normalization dimensions.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth parameter.
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix.
    metric : str, optional
        Metric to use for pairwise distances computation.
    nodiag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
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
        nodiag: bool = True,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric, nodiag=nodiag, device=device, keops=keops, verbose=verbose
        )
        self.sigma = sigma
        self.normalization_dim = normalization_dim

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the Gibbs affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data.

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
            self.log_affinity_matrix_ = (
                self.log_affinity_matrix_ - self.log_normalization_
            )

        return self

    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Extracts the affinity submatrix corresponding to the indices.

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

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()


class LocalGibbsAffinity(LogAffinity):
    r"""
    Computes the Gibbs affinity matrix
    :math:`\exp( - \mathbf{C} / \mathbf{\sigma} \mathbf{\sigma}^\top)` with
    sample-wise bandwidth :math:`\mathbf{\sigma} \in \R^n` based on the
    neirest neighbor strategy of [22]_, where :math:`\mathbf{C}` is the
    pairwise distance matrix.

    Parameters
    ----------
    K : int, optional
        K-th neirest neighbor .
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix.
    metric : str, optional
        Metric to use for pairwise distances computation.
    nodiag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity.

    References
    ----------
    .. [22] Max Zelnik-Manor, L., & Perona, P. (2004).
            Self-tuning spectral clustering. Advances in neural information
            processing systems (NIPS).
    """

    def __init__(
        self,
        K: int = 7,
        normalization_dim: int | Tuple[int] = (0, 1),
        metric: str = "euclidean",
        nodiag: bool = True,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric, nodiag=nodiag, device=device, keops=keops, verbose=verbose
        )
        self.K = K
        self.normalization_dim = normalization_dim

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the local Gibbs affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data.

        Returns
        -------
        self : LocalGibbsAffinity
            The fitted local Gibbs affinity model.
        """
        super().fit(X)
        C = self._pairwise_distance_matrix(self.data_)

        minK_values, minK_indices = kmin(C, k=self.K, dim=1)
        self.sigma_ = minK_values[:, -1]
        self.log_affinity_matrix_ = _log_LocalGibbs(C, self.sigma_)

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                self.log_affinity_matrix_, self.normalization_dim
            )
            self.log_affinity_matrix_ = (
                self.log_affinity_matrix_ - self.log_normalization_
            )

        return self

    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Extracts the affinity submatrix corresponding to the indices.

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
        log_P_batch = _log_LocalGibbs(C_batch, self.sigma_)

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()


class StudentAffinity(LogAffinity):
    r"""
    Computes the Student affinity matrix based on the Student-t distribution:

    .. math::
        \left(1 + \frac{\mathbf{C}}{\nu}\right)^{-\frac{\nu + 1}{2}}

    where :math:`\nu > 0` is the degrees of freedom parameter.

    Parameters
    ----------
    degrees_of_freedom : int, optional
        Degrees of freedom for the Student-t distribution.
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix.
    metric : str, optional
        Metric to use for pairwise distances computation.
    nodiag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
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
        nodiag: bool = True,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric, nodiag=nodiag, device=device, keops=keops, verbose=verbose
        )
        self.normalization_dim = normalization_dim
        self.degrees_of_freedom = degrees_of_freedom

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the Student affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data.

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
            self.log_affinity_matrix_ = (
                self.log_affinity_matrix_ - self.log_normalization_
            )

        return self

    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Extracts the affinity submatrix corresponding to the indices.

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
            log_P_batch = log_P_batch - log_normalization_batch

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()
