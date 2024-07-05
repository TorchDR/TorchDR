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
from torchdr.utils import (
    symmetric_pairwise_distances,
    symmetric_pairwise_distances_indices,
    pairwise_distances,
)


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
        self.zero_diag = zero_diag
        self.add_diagonal = 1e12 if self.zero_diag else None

    @abstractmethod
    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Prepares and stores the input data :math:`\mathbf{X}` for computing
        the affinity matrix.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data to be converted and stored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : fit method not implemented for affinity "
            f"{self.__class__.__name__}. "
        )

    def fit_transform(self, X: torch.Tensor | np.ndarray):
        r"""
        Computes the affinity matrix from input data :math:`\mathbf{X}` and returns
        the resulting matrix.

        It first calls the `fit` method to compute the affinity matrix from
        the input data and then returns the computed affinity matrix.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
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
            msg="[TorchDR] ERROR : affinity_matrix_ should be computed in fit method."
        )
        return self.affinity_matrix_  # type: ignore

    def _distance_matrix(self, X: torch.Tensor):
        r"""
        Computes the pairwise distance matrix :math:`\mathbf{C}` for the input tensor.

        This method calculates the pairwise distances between all samples in the input
        tensor :math:`\mathbf{X}`, using the specified metric and optionally leveraging
        KeOps for (memory) efficient computation.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : torch.Tensor or pykeops.torch.LazyTensor
            The pairwise distance matrix. The type of the returned matrix depends on the
            value of the `keops` attribute. If `keops` is True, a KeOps LazyTensor
            is returned. Otherwise, a torch.Tensor is returned.
        """
        C = symmetric_pairwise_distances(
            X=X,
            metric=self.metric,
            keops=self.keops,
            add_diagonal=self.add_diagonal,
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
            msg or "[TorchDR] ERROR : Affinity not fitted."
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
        C_batch : torch.Tensor or pykeops.torch.LazyTensor of shape
        (n_batch, batch_size, batch_size)
            The batched pairwise distance matrix.
        """
        self._check_is_fitted()
        assert (
            indices.ndim == 2
        ), '[TorchDR] ERROR : indices in "get_batch" should be a 2D torch tensor '
        "of shape (n_batch, batch_size)."
        assert (
            indices.shape[0] * indices.shape[1] == self.data_.shape[0]
        ), '[TorchDR] ERROR : indices in "get_batch" should have a product '
        "of dimensions equal to the number of samples."

        data_batch = self.data_[indices]
        C_batch = self._distance_matrix(data_batch)
        return C_batch

    def transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        indices: torch.Tensor = None,
    ):
        r"""
        Computes the affinity between points without fitting any parameter.
        Thus it can only be called for affinities that do not require any fitting.
        For such affinities, this method must be overridden.

        Raises
        ------
        NotImplementedError
            If the method is called for an affinity that requires fitting, an error
            is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : transform method not implemented for affinity "
            f"{self.__class__.__name__}. This means that the affinity has normalizing "
            "parameters that need to be fitted. Thus it can only be called using the "
            "fit_transform method."
        )

    def _distance_matrix_transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        indices: torch.Tensor = None,
    ):
        r"""
        Computes the pairwise distance matrix between two datasets X and Y (useful for
        the transform method of kernel PCA) or between a dataset X and itself (regular
        usecase of dimensionality reduction).
        If indices is provided, computes pairwise distances between X and iteself
        for a subset of pairs given by indices.
        """

        if Y is not None and indices is not None:
            raise NotImplementedError(
                "[TorchDR] ERROR : transform method cannot be called with both Y "
                "and indices at the same time."
            )

        if indices is not None:
            return symmetric_pairwise_distances_indices(
                X, indices=indices, metric=self.metric
            )

        elif Y is not None:
            return pairwise_distances(X, Y, metric=self.metric, keops=self.keops)

        else:
            return symmetric_pairwise_distances(
                X, metric=self.metric, keops=self.keops, add_diagonal=self.add_diagonal
            )

    def check_transform_implemented(self):
        """
        Checks if the transform method has been implemented (must be overridden and
        not just the base class).
        """
        # Check if the method is overridden and not just the base class method
        method = getattr(self, "transform")
        if method.__func__ is not Affinity.transform:
            return True
        else:
            return False


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
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data used to compute the affinity matrix.

        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.

        Returns
        -------
        affinity_matrix_ : torch.Tensor or pykeops.torch.LazyTensor of shape
        (n_samples, n_samples)
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
        ), "[TorchDR] ERROR : log_affinity_matrix_ should be computed "
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
            msg or "[TorchDR] ERROR : LogAffinity not fitted."
        )
