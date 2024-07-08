# -*- coding: utf-8 -*-
"""
Base classes for affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod

import torch
try :  # try to import LazyTensor from KeOps for type hinting
    from keops.torch import LazyTensor
except ImportError:
    LazyTensor = type(None)

import numpy as np
from torchdr.utils import (
    symmetric_pairwise_distances,
    symmetric_pairwise_distances_indices,
    pairwise_distances,
    to_torch,
    inputs_to_torch,
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
        Fits the affinity matrix from the input data.
        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data to be converted and stored.
        """
        pass

    def fit_transform(self, X: torch.Tensor | np.ndarray):
        r"""
        Computes the affinity matrix from the input data using the `fit` method and
        returns the resulting affinity matrix.

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

        if not hasattr(self, "affinity_matrix_"):
            raise AssertionError(
                "[TorchDR] ERROR : affinity_matrix_ should be computed in fit method."
            )

        return self.affinity_matrix_

    def _distance_matrix(self, X: torch.Tensor):
        r"""
        Computes the pairwise distance matrix from the input data.

        This method calculates the pairwise distances between all samples in the input
        data, using the specified metric and optionally leveraging KeOps for memory
        efficient computation.

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

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second input data. If None, uses `Y=X`. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. Default is None.

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
        if Y is not None and indices is not None:
            raise NotImplementedError(
                "[TorchDR] ERROR : transform method cannot be called with both Y "
                "and indices at the same time."
            )

        elif indices is not None:
            return symmetric_pairwise_distances_indices(
                X, indices=indices, metric=self.metric
            )

        elif Y is not None:
            return pairwise_distances(X, Y, metric=self.metric, keops=self.keops)

        else:
            return symmetric_pairwise_distances(
                X, metric=self.metric, keops=self.keops, add_diagonal=self.add_diagonal
            )


class TransformableAffinity(Affinity):
    r"""
    Base class for affinities that do not require fitting parameters based on
    the entire dataset. These affinities can be applied directly to a subset
    of the data using the `transform` method.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
        Default is "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero. Default is True.
    device : str, optional
        The device to use for computation, e.g., "cuda" for GPU or "cpu" for CPU.
        If "auto", it uses the device of the input data. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel
        operations. Default is False.
    verbose : bool, optional
        If True, prints additional information during computation. Default is True.
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

    @abstractmethod
    def _affinity_formula(self, C: torch.Tensor | LazyTensor):
        r"""
        Computes the affinity matrix from the pairwise distance matrix.
        This method must be overridden by subclasses.

        Parameters
        ----------
        C : torch.Tensor or pykeops.torch.LazyTensor
            Pairwise distance matrix.
        """
        pass

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data.

        Returns
        -------
        self : TransformableAffinity
            The fitted affinity model.
        """
        self.data_ = to_torch(X, device=self.device, verbose=self.verbose)
        C = self._distance_matrix(self.data_)
        self.affinity_matrix_ = self._affinity_formula(C)
        return self

    @inputs_to_torch
    def transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        indices: torch.Tensor = None,
    ):
        r"""
        Computes the affinity between points without fitting any parameters.
        Suitable for affinities that do not require normalization based on the
        entire dataset.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second input data. If None, uses `Y=X`. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. If None, computes the full affinity matrix.
            Default is None.

        Returns
        -------
        P : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix or its log, depending on the `log` parameter.
        """
        C = self._distance_matrix_transform(X, Y=Y, indices=indices)
        return self._affinity_formula(C)


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
        Computes the log affinity matrix from the input data using the `fit` method
        and returns the resulting affinity matrix.

        It returns either the log affinity matrix or the exponential of the log
        affinity matrix, depending on the value of the `log` parameter.

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

        if not hasattr(self, "log_affinity_matrix_"):
            raise AssertionError(
                "[TorchDR] ERROR : log_affinity_matrix_ should be computed in the "
                "fit method of a LogAffinity."
            )

        if log:  # return the log of the affinity matrix
            return self.log_affinity_matrix_  # type: ignore
        else:
            if not hasattr(self, "affinity_matrix_"):
                self.affinity_matrix_ = self.log_affinity_matrix_.exp()  # type: ignore
            return self.affinity_matrix_


class TransformableLogAffinity(LogAffinity):
    r"""
    Base class for log affinities that do not require fitting parameters based on
    the entire dataset. These affinities can be applied directly to a subset
    of the data using the `transform` method.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
        Default is "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero. Default is True.
    device : str, optional
        The device to use for computation, e.g., "cuda" for GPU or "cpu" for CPU.
        If "auto", it uses the device of the input data. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel
        operations. Default is False.
    verbose : bool, optional
        If True, prints additional information during computation. Default is True.
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

    @abstractmethod
    def _log_affinity_formula(self, C: torch.Tensor | LazyTensor):
        r"""
        Computes the log affinity matrix from the pairwise distances.
        This method must be overridden by subclasses.

        Parameters
        ----------
        C : torch.Tensor or pykeops.torch.LazyTensor
            Pairwise distance matrix.
        """
        pass

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data.

        Returns
        -------
        self : TransformableLogAffinity
            The fitted log affinity model.
        """
        self.data_ = to_torch(X, device=self.device, verbose=self.verbose)
        C = self._distance_matrix(self.data_)
        self.log_affinity_matrix_ = self._log_affinity_formula(C)
        return self

    @inputs_to_torch
    def transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        log: bool = False,
        indices: torch.Tensor = None,
    ):
        r"""
        Computes the log affinity between points without fitting any parameters.
        Suitable for affinities that do not require normalization based on the
        entire dataset.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second input data. If None, uses `Y=X`. Default is None.
        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. If None, computes the full affinity matrix.
            Default is None.

        Returns
        -------
        P : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix or its log, depending on the `log` parameter.
        """
        C = self._distance_matrix_transform(X, Y=Y, indices=indices)
        log_P = self._log_affinity_formula(C)

        if log:
            return log_P
        else:
            return log_P.exp()


class SparseLogAffinity(LogAffinity):
    r"""
    Base class for sparse log affinity matrices.

    Modifies the fit_transform method of the base LogAffinity class to return the log
    affinity matrix in a rectangular format with the corresponding indices.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
        Default is "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero. Default is True.
    device : str, optional
        The device to use for computation. Typically "cuda" for GPU or "cpu" for CPU.
        If "auto", uses the device of the input data. Default is "auto".
    keops : bool, optional
        Whether to use KeOps for efficient computation of large-scale kernel
        operations. Default is False.
    verbose : bool, optional
        If True, prints additional information during computation. Default is True.
    sparsity : bool or str, optional
        Whether to compute the affinity matrix in a sparse format. Default is "auto".
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
        sparsity: bool | str = "auto",
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        self.sparsity = sparsity
        if sparsity == "auto":
            self._sparsity = self._sparsity_rule()
        else:
            self._sparsity = sparsity

    @abstractmethod
    def _sparsity_rule(self):
        r"""
        Rule to determine whether to compute the affinity matrix in a sparse format.
        This method must be overridden by subclasses.
        """
        pass

    def fit_transform(self, X: torch.Tensor | np.ndarray, log: bool = False):
        r"""
        Computes the log affinity matrix from input data using the `fit` method and
        returns the resulting log affinity matrix.

        If sparsity is enabled, returns the log affinity in rectangular format with the
        corresponding indices. Otherwise, returns the full affinity matrix and None.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data used to compute the affinity matrix.
        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.

        Returns
        -------
        log_affinity_matrix_ : torch.Tensor or pykeops.torch.LazyTensor
            The computed log affinity matrix if `log` is True, otherwise the
            exponentiated affinity matrix.
        indices_ : torch.Tensor
            The indices of the non-zero elements in the affinity matrix if sparsity is
            enabled. Otherwise, returns None.

        Raises
        ------
        AssertionError
            If the `log_affinity_matrix_` or `indices_` attribute is not set during the
            `fit` method, an assertion error is raised.
        """
        self.fit(X)

        if not hasattr(self, "log_affinity_matrix_") or not hasattr(self, "indices_"):
            raise AssertionError(
                "[TorchDR] ERROR : log_affinity_matrix_ and indices_ should be "
                "computed in the fit method of a SparseLogAffinity."
            )

        if log:
            return self.log_affinity_matrix_, self.indices_
        else:
            return self.log_affinity_matrix_.exp(), self.indices_
