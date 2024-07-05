# -*- coding: utf-8 -*-
"""
Common simple affinities
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from typing import Tuple

from torchdr.utils import logsumexp_red, to_torch
from torchdr.affinity.base import Affinity, LogAffinity
from torchdr.utils import (
    extract_batch_normalization,
    output_exp_if_not_log,
    inputs_to_torch,
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
        device: str = "cuda",
        keops: bool = False,
        verbose: bool = True,
        centering: bool = False,
    ):
        super().__init__(
            metric="angular", device=device, keops=keops, verbose=verbose, nodiag=False
        )
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
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : ScalarProductAffinity
            The fitted scalar product affinity model.
        """
        self.data_ = to_torch(X, device=self.device, verbose=self.verbose)
        if self.centering:
            self.data_ = self.data_ - self.data_.mean(0)
        self.affinity_matrix_ = -self._distance_matrix(self.data_)

        return self

    def get_batch(self, indices: torch.Tensor):
        r"""
        Extracts the fitted affinity submatrix corresponding to the indices.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        P_batch : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_batch, batch_size, batch_size)
            The affinity matrix for the batch indices.
        """
        C_batch = super().get_batch(indices)
        return -C_batch

    @inputs_to_torch
    def transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        indices: torch.Tensor = None,
    ):
        r"""
        Computes the scalar product affinity between X and Y.
        If Y is None, sets Y = X.
        If indices is not None, the output has shape (n, k) and its (i,j) element is the
        affinity between X[i] and Y[indices[i, j]].

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second Input data. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. Default is None.

        Returns
        -------
        P : torch.Tensor or pykeops.torch.LazyTensor
            Scalar product between X and Y.
        """
        C = self._distance_matrix_transform(X, Y=Y, indices=indices)
        return -C


class GibbsAffinity(LogAffinity):
    r"""
    Computes the Gibbs affinity matrix :math:`\exp( - \mathbf{C} / \sigma)`
    where :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma` is the bandwidth parameter.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth parameter.
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
        metric: str = "sqeuclidean",
        nodiag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric, nodiag=nodiag, device=device, keops=keops, verbose=verbose
        )
        self.sigma = sigma

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
        self.data_ = to_torch(X, device=self.device, verbose=self.verbose)
        C = self._distance_matrix(self.data_)
        self.log_affinity_matrix_ = _log_Gibbs(C, self.sigma)
        return self

    @output_exp_if_not_log
    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Extracts the fitted affinity submatrix corresponding to the indices.

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
        return log_P_batch

    @output_exp_if_not_log
    @inputs_to_torch
    def transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        log: bool = False,
        indices: torch.Tensor = None,
    ):
        r"""
        Computes the Gibbs affinity between X and Y.
        If Y is None, sets Y = X.
        If indices is not None, the output has shape (n, k) and its (i,j) element is the
        affinity between X[i] and Y[indices[i, j]].

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second Input data. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. Default is None.

        Returns
        -------
        P : torch.Tensor or pykeops.torch.LazyTensor
            Scalar product between X and Y.
        """
        C = self._distance_matrix_transform(X, Y=Y, indices=indices)
        return _log_Gibbs(C, self.sigma)


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
        metric: str = "sqeuclidean",
        nodiag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric, nodiag=nodiag, device=device, keops=keops, verbose=verbose
        )
        self.degrees_of_freedom = degrees_of_freedom

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the Student affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : StudentAffinity
            The fitted Student affinity model.
        """
        self.data_ = to_torch(X, device=self.device, verbose=self.verbose)
        C = self._distance_matrix(self.data_)
        self.log_affinity_matrix_ = _log_Student(C, self.degrees_of_freedom)
        return self

    @output_exp_if_not_log
    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Extracts the fitted affinity submatrix corresponding to the indices.

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
        return log_P_batch

    @output_exp_if_not_log
    @inputs_to_torch
    def transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        log: bool = False,
        indices: torch.Tensor = None,
    ):
        r"""
        Computes the Student affinity between X and Y.
        If Y is None, sets Y = X.
        If indices is not None, the output has shape (n, k) and its (i,j) element is the
        affinity between X[i] and Y[indices[i, j]].

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second Input data. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. Default is None.

        Returns
        -------
        P : torch.Tensor or pykeops.torch.LazyTensor
            Scalar product between X and Y.
        """
        C = self._distance_matrix_transform(X, Y=Y, indices=indices)
        return _log_Student(C, self.degrees_of_freedom)


class NormalizedGibbsAffinity(GibbsAffinity):
    r"""
    Computes the Gibbs affinity matrix :math:`\exp( - \mathbf{C} / \sigma)`
    where :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma` is the bandwidth parameter. The affinity can be normalized
    according to the specified normalization dimensions.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth parameter.
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
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        metric: str = "sqeuclidean",
        nodiag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
        normalization_dim: int | Tuple[int] = (0, 1),
    ):
        super().__init__(
            sigma=sigma,
            metric=metric,
            nodiag=nodiag,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        self.normalization_dim = normalization_dim

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the normalized Gibbs affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : GibbsAffinity
            The fitted Gibbs affinity model.
        """
        super().fit(X)

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                self.log_affinity_matrix_, self.normalization_dim
            )
            self.log_affinity_matrix_ = (
                self.log_affinity_matrix_ - self.log_normalization_
            )

        return self

    @output_exp_if_not_log
    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Extracts the fitted affinity submatrix corresponding to the indices.

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
        log_P_batch = super().get_batch(indices, log=True)

        if self.normalization_dim is not None:
            log_normalization_batch = extract_batch_normalization(
                self.log_normalization_, indices, self.normalization_dim
            )
            log_P_batch = log_P_batch - log_normalization_batch

        return log_P_batch
