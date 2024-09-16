# -*- coding: utf-8 -*-
"""Base classes for affinity matrices."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC

import torch

import numpy as np
from torchdr.utils import (
    symmetric_pairwise_distances,
    symmetric_pairwise_distances_indices,
    pairwise_distances,
    to_torch,
    LazyTensorType,
    pykeops,
)


class Affinity(ABC):
    r"""Base class for affinity matrices.

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
        If True, prints additional information during computation. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
    ):
        if keops and not pykeops:
            raise ValueError(
                "[TorchDR] ERROR : pykeops is not installed. Please install it to use "
                "`keops=true`."
            )

        self.log = {}
        self.metric = metric
        self.zero_diag = zero_diag
        self.device = device
        self.keops = keops
        self.verbose = verbose
        self.zero_diag = zero_diag
        self.add_diag = 1e12 if self.zero_diag else None

    def __call__(self, X: torch.Tensor | np.ndarray, **kwargs):
        r"""Compute the affinity matrix from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix.
        """
        X = to_torch(X, device=self.device)
        return self._compute_affinity(X, **kwargs)

    def _compute_affinity(self, X: torch.Tensor):
        r"""Compute the affinity matrix from the input data.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Raises
        ------
        NotImplementedError
            If the `_compute_affinity` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_affinity` method is not implemented."
        )

    def _distance_matrix(self, X: torch.Tensor):
        r"""Compute the pairwise distance matrix from the input data.

        It uses the specified metric and optionally leveraging KeOps
        for memory efficient computation.

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
        return symmetric_pairwise_distances(
            X=X,
            metric=self.metric,
            keops=self.keops,
            add_diag=self.add_diag,
        )


class LogAffinity(Affinity):
    r"""Base class for affinity matrices in log domain.

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
        If True, prints additional information during computation. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )

    def __call__(self, X: torch.Tensor | np.ndarray, log: bool = False, **kwargs):
        r"""Compute the affinity matrix from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.
        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed log affinity matrix if `log` is True, otherwise the
            exponentiated log affinity matrix.
        """
        X = to_torch(X, device=self.device)
        log_affinity = self._compute_log_affinity(X, **kwargs)
        if log:
            return log_affinity
        else:
            return log_affinity.exp()

    def _compute_log_affinity(self, X: torch.Tensor):
        r"""Compute the log affinity matrix from the input data.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Raises
        ------
        NotImplementedError
            If the `_compute_log_affinity` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_log_affinity` method is not implemented."
        )


class SparseLogAffinity(LogAffinity):
    r"""Base class for sparse log affinity matrices.

    If sparsity is enabled, returns the log affinity matrix in a rectangular format
    with the corresponding indices.
    Otherwise, returns the full affinity matrix and None.

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
        If True, prints additional information during computation. Default is False.
    sparsity : bool or 'auto', optional
        Whether to compute the affinity matrix in a sparse format. Default is "auto".
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
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

    def _sparsity_rule(self):
        r"""Rule to determine whether to compute the affinity matrix in a sparse format.

        This method must be overridden by subclasses.

        Raises
        ------
        NotImplementedError
            If the `_sparsity_rule` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_sparsity_rule` method is not implemented. "
            "Therefore sparsity = 'auto' is not supported."
        )

    def __call__(
        self,
        X: torch.Tensor | np.ndarray,
        log: bool = False,
        return_indices: bool = False,
        **kwargs,
    ):
        r"""Compute and return the log affinity matrix from input data.

        If sparsity is enabled, returns the log affinity in rectangular format with the
        corresponding indices. Otherwise, returns the full affinity matrix and None.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data used to compute the affinity matrix.
        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.
        return_indices : bool, optional
            If True, returns the indices of the non-zero elements in the affinity matrix
            if sparsity is enabled. Default is False.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed log affinity matrix if `log` is True, otherwise the
            exponentiated log affinity matrix.
        indices : torch.Tensor
            If return_indices is True, returns the indices of the non-zero elements
            in the affinity matrix if sparsity is enabled. Otherwise, returns None.
        """
        X = to_torch(X, device=self.device)
        log_affinity, indices = self._compute_sparse_log_affinity(X, **kwargs)
        affinity_to_return = log_affinity if log else log_affinity.exp()
        return (affinity_to_return, indices) if return_indices else affinity_to_return

    def _compute_sparse_log_affinity(self, X: torch.Tensor):
        r"""Compute the log affinity matrix in a sparse format from the input data.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.

        Raises
        ------
        NotImplementedError
            If the `_compute_sparse_log_affinity` method is not implemented by
            the subclass, a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_sparse_log_affinity` method is "
            "not implemented."
        )


class UnnormalizedAffinity(Affinity):
    r"""Base class for unnormalized affinities.

    These affinities are defined using a closed-form formula on the pairwise distance
    matrix and can be directly applied to a subset of the data by providing indices.

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
        If True, prints additional information during computation. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )

    def __call__(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        indices: torch.Tensor = None,
        **kwargs,
    ):
        r"""Compute the affinity matrix from the input data.

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
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix.
        """
        X = to_torch(X, device=self.device)
        if Y is not None:
            Y = to_torch(Y, device=self.device)
        C = self._distance_matrix(X=X, Y=Y, indices=indices, **kwargs)
        return self._affinity_formula(C)

    def _affinity_formula(self, C: torch.Tensor | LazyTensorType):
        r"""Compute the affinity from the distance matrix.

        This method must be overridden by subclasses.

        Parameters
        ----------
        C : torch.Tensor or pykeops.torch.LazyTensor
            Pairwise distance matrix.

        Raises
        ------
        NotImplementedError
            If the `_affinity_formula` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_affinity_formula` method is not implemented."
        )

    def _distance_matrix(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        indices: torch.Tensor = None,
    ):
        r"""Compute the pairwise distance matrix from the input data.

        It uses the specified metric and optionally leverages KeOps
        for memory efficient computation.
        It supports computing the full pairwise distance matrix, the pairwise
        distance matrix between two sets of samples, or the pairwise distance matrix
        between a set of samples and a subset of samples specified by indices.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second input data. If None, uses `Y=X`. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. If None, computes the full pairwise distance
            matrix. Default is None.

        Returns
        -------
        C : torch.Tensor or pykeops.torch.LazyTensor
            The pairwise distance matrix. The type of the returned matrix depends on the
            value of the `keops` attribute. If `keops` is True, a KeOps LazyTensor
            is returned. Otherwise, a torch.Tensor is returned.
        """
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
                X, metric=self.metric, keops=self.keops, add_diag=self.add_diag
            )


class UnnormalizedLogAffinity(UnnormalizedAffinity):
    r"""Base class for unnormalized affinities in log domain.

    These log affinities are defined using a closed-form formula on the pairwise
    distance matrix and can be directly applied to a subset of the data by providing
    indices.

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
        If True, prints additional information during computation. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )

    def __call__(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        indices: torch.Tensor = None,
        log: bool = False,
        **kwargs,
    ):
        r"""Compute the affinity matrix in log domain from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second input data. If None, uses `Y=X`. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. If None, computes the full affinity matrix.
            Default is None.
        log : bool, optional
            If True, returns the log of the affinity matrix. Else, returns
            the affinity matrix by exponentiating the log affinity matrix.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix.
        """
        X = to_torch(X, device=self.device)
        if Y is not None:
            Y = to_torch(Y, device=self.device)
        C = self._distance_matrix(X=X, Y=Y, indices=indices, **kwargs)
        log_affinity = self._log_affinity_formula(C)
        if log:
            return log_affinity
        else:
            return log_affinity.exp()

    def _log_affinity_formula(self, C: torch.Tensor | LazyTensorType):
        r"""Compute the log affinity from the distance matrix.

        This method must be overridden by subclasses.

        Parameters
        ----------
        C : torch.Tensor or pykeops.torch.LazyTensor
            Pairwise distance matrix.

        Raises
        ------
        NotImplementedError
            If the `_log_affinity_formula` method is not implemented by the subclass,
            a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_log_affinity_formula` method is not implemented."
        )
