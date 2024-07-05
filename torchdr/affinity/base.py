# -*- coding: utf-8 -*-
"""
Base classes for affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod

import torch
import numpy as np
from torchdr.utils import pairwise_distances, to_torch


class Affinity(ABC):
    r"""
    Base class for affinity matrices.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        The device to use for computation. Typically "cuda" for GPU or "cpu" for CPU.
        If "auto", uses the device of the input data.
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel operations.
    verbose : bool, optional
        If True, prints additional information during computation (default is True).
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
    ):
        self.log = {}
        self.metric = metric
        self.zero_diag = zero_diag
        self.device = device
        self.keops = keops
        self.verbose = verbose

    @abstractmethod
    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Prepares and stores the input data :math:`\mathbf{X}` for computing
        the affinity matrix.

        This method must be overridden by subclasses. This base implementation
        only converts the input data to a torch tensor and stores it
        in the `data_` attribute.

        Subclasses should call `super().fit(X)` to utilize this functionality
        and then implement additional steps required for computing the specific
        affinity matrix.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data to be converted and stored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.data_ = to_torch(X, device=self.device, verbose=self.verbose)
        return self

    def fit_transform(self, X: torch.Tensor | np.ndarray):
        r"""
        Computes the affinity matrix from input data :math:`\mathbf{X}` and returns
        the resulting matrix.

        It first calls the `fit` method to compute the affinity matrix from
        the input data and then returns the computed affinity matrix.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data used to compute the affinity matrix.

        Returns
        -------
        affinity_matrix_ : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix.

        Raises
        ------
        AssertionError
            If the `affinity_matrix_` attribute is not set during the `fit` method,
            an assertion error is raised.
        """
        self.fit(X)
        self._check_is_fitted(
            msg="[TorchDR] Error : affinity_matrix_ should be computed in fit method."
        )
        return self.affinity_matrix_  # type: ignore

    def _pairwise_distance_matrix(self, X: torch.Tensor, Y: torch.Tensor = None):
        r"""
        Computes the pairwise distance matrix :math:`\mathbf{C}` for the input tensor.

        This method calculates the pairwise distances between all samples in the input
        tensor :math:`\mathbf{X}`, using the specified metric and optionally leveraging
        KeOps for (memory) efficient computation.

        Parameters
        ----------
        X : torch.Tensor
            A 2D tensor of shape (n_samples, n_features) containing the input data.

        Returns
        -------
        C : torch.Tensor or pykeops.torch.LazyTensor
            The pairwise distance matrix. The type of the returned matrix depends on the
            value of the `keops` attribute. If `keops` is True, a KeOps LazyTensor
            is returned. Otherwise, a torch.Tensor is returned.
        """
        add_diagonal = 1e12 if self.zero_diag else None
        C = pairwise_distances(
            X, Y, metric=self.metric, keops=self.keops, add_diagonal=add_diagonal
        )
        return C

    def _check_is_fitted(self, msg: str = None):
        r"""
        Checks if the affinity matrix has been computed and is stored in
        the attribute affinity_matrix_ .

        Parameters
        ----------
        msg : str, optional
            Custom error message to be displayed if the check fails. If not provided,
            a default error message is used.

        Raises
        ------
        AssertionError
            If the `affinity_matrix_` attribute does not exist, indicating that
            the model has not been fitted.
        """
        assert hasattr(self, "affinity_matrix_"), (
            msg or "[TorchDR] Error : Affinity not fitted."
        )

    @abstractmethod
    def get_batch(self, indices: torch.Tensor):
        r"""
        Decomposes the fitted affinity into batches based on the provided indices.

        This method must be overridden by subclasses. This base implementation returns
        the batched pairwise distance matrix. Subclasses should call
        `super().get_batch(indices)` as a first step to get the batched distance matrix
        and then implement additional steps to compute the affinity.

        The total number of samples must equal the product of the number of batches
        and the batch size.

        Parameters
        ----------
        indices : torch.Tensor
            A 2D tensor of shape (n_batch, batch_size) containing the batch indices.
            The number of samples must equal the product of n_batch and batch_size.

        Returns
        -------
        C_batch : torch.Tensor or pykeops.torch.LazyTensor
            The batched pairwise distance matrix.
        """
        self._check_is_fitted()
        assert (
            indices.ndim == 2
        ), '[TorchDR] Error: indices in "get_batch" should be a 2D torch tensor '
        "of shape (n_batch, batch_size)."
        assert (
            indices.shape[0] * indices.shape[1] == self.data_.shape[0]
        ), '[TorchDR] Error: indices in "get_batch" should have a product '
        "of dimensions equal to the number of samples."

        data_batch = self.data_[indices]
        C_batch = self._pairwise_distance_matrix(data_batch)
        return C_batch


class LogAffinity(Affinity):
    r"""
    Base class for affinity matrices in log space.

    This class inherits from the `Affinity` base class and is designed to work with
    affinity matrices in log space. It provides methods to fit the model to input
    data and transform it to an affinity matrix, optionally in log space.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
    device : str, optional
        The device to use for computation. Typically "cuda" for GPU or "cpu" for CPU.
        If "auto", uses the device of the input data.
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel operations.
    verbose : bool, optional
        If True, prints additional information during computation (default is True).
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )

    def fit_transform(self, X: torch.Tensor | np.ndarray, log: bool = False):
        r"""
        Fits the model to the input data :math:`\mathbf{X}` and returns
        the affinity matrix.

        This method first calls the `fit` method to compute the log affinity matrix
        from the input data. It then returns either the log affinity matrix or the
        exponential of the log affinity matrix, depending on the value of
        the `log` parameter.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data used to compute the affinity matrix.

        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.

        Returns
        -------
        affinity_matrix_ : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix. If `log` is True, returns the log affinity
            matrix. Otherwise, returns the exponentiated log affinity matrix.

        Raises
        ------
        AssertionError
            If the `log_affinity_matrix_` attribute is not set during the `fit` method,
            an assertion error is raised.
        """
        self.fit(X)
        assert hasattr(
            self, "log_affinity_matrix_"
        ), "[TorchDR] ERROR Affinity : log_affinity_matrix_ should be computed "
        "in  fit method of a LogAffinity."

        if log:  # return the log of the affinity matrix
            return self.log_affinity_matrix_  # type: ignore
        else:
            if not hasattr(self, "affinity_matrix_"):
                self.affinity_matrix_ = self.log_affinity_matrix_.exp()  # type: ignore
            return self.affinity_matrix_

    def _check_is_fitted(self, msg: str = None):
        r"""
        Checks if the log affinity matrix has been computed and is stored in
        the attribute log_affinity_matrix_ .

        Parameters
        ----------
        msg : str, optional
            Custom error message to be displayed if the check fails. If not provided,
            a default error message is used.

        Raises
        ------
        AssertionError
            If the `log_affinity_matrix_` attribute does not exist, indicating that
            the model has not been fitted.
        """
        assert hasattr(self, "log_affinity_matrix_"), (
            msg or "[TorchDR] Error : LogAffinity not fitted."
        )
