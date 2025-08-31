"""Base classes for affinity matrices."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC
from typing import Union, Any

import numpy as np
import torch
import torch.nn as nn

from torchdr.utils import (
    to_torch,
    bool_arg,
    set_logger,
)

from torchdr.distance import (
    pairwise_distances,
    FaissConfig,
)


class Affinity(nn.Module, ABC):
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
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
          (e.g., FaissConfig(use_float16=True, temp_memory=2.0))
        Default is None.
    verbose : bool, optional
        Verbosity. Default is False.
    compile : bool, optional
        Whether to compile the affinity matrix computation. Default is False.
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Union[str, FaissConfig] = None,
        verbose: bool = False,
        random_state: float = None,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        super().__init__()

        self.log = {}
        self.metric = metric
        self.zero_diag = bool_arg(zero_diag)
        self.device = device
        self.backend = backend
        self.verbose = bool_arg(verbose)
        self.random_state = random_state
        self.compile = compile
        self._pre_processed = _pre_processed

        self.logger = set_logger(self.__class__.__name__, self.verbose)

    def __call__(self, X: Union[torch.Tensor, np.ndarray], **kwargs):
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
        if not self._pre_processed:
            X = to_torch(X)
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

    def _distance_matrix(
        self, X: torch.Tensor, k: int = None, return_indices: bool = False
    ):
        r"""Compute the pairwise distance matrix from the input data.

        It uses the specified metric and optionally leveraging KeOps
        for memory efficient computation.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.
        k : int, optional
            Number of nearest neighbors to compute the distance matrix. Default is None.
        return_indices : bool, optional
            Whether to return the indices of the k-nearest neighbors.
            Default is False.

        Returns
        -------
        C : torch.Tensor or pykeops.torch.LazyTensor
            The pairwise distance matrix. The type of the returned matrix depends on the
            value of the `backend` attribute. If `backend` is `keops`, a KeOps LazyTensor
            is returned. Otherwise, a torch.Tensor is returned.
        """
        return pairwise_distances(
            X=X,
            metric=self.metric,
            backend=self.backend,
            exclude_diag=self.zero_diag,  # infinite distance means zero affinity
            k=k,
            return_indices=return_indices,
            device=self.device,  # Pass computation device (can be "auto")
        )

    def clear_memory(self):
        """Clear non-persistent buffers to free memory."""
        if hasattr(self, "_non_persistent_buffers_set"):
            for name in list(self._non_persistent_buffers_set):
                if hasattr(self, name):
                    delattr(self, name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LogAffinity(Affinity):
    r"""Base class for affinity matrices in log domain.

    Parameters
    ----------
    metric : str, optional
        The distance metric to use for computing pairwise distances.
    device : str, optional
        The device to use for computation. Typically "cuda" for GPU or "cpu" for CPU.
        If "auto", uses the device of the input data.
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
        Default is None.
    verbose : bool, optional
        If True, prints additional information during computation. Default is False.
    compile : bool, optional
        Whether to compile the affinity matrix computation. Default is False.
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Union[str, FaissConfig] = None,
        verbose: bool = False,
        random_state: float = None,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            compile=compile,
            _pre_processed=_pre_processed,
        )

    def __call__(
        self, X: Union[torch.Tensor, np.ndarray], log: bool = False, **kwargs: Any
    ):
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
        if not self._pre_processed:
            X = to_torch(X)
        log_affinity = self._compute_log_affinity(X, **kwargs)
        if log:
            return log_affinity
        else:
            return log_affinity.exp()

    def _compute_log_affinity(self, X: torch.Tensor, **kwargs):
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


class SparseAffinity(Affinity):
    r"""Base class for sparse affinity matrices.

    If sparsity is enabled, returns the affinity matrix in a rectangular format
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
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
        Default is None.
    verbose : bool, optional
        If True, prints additional information during computation. Default is False.
    compile : bool, optional
        Whether to compile the affinity matrix computation. Default is False.
    sparsity : bool or 'auto', optional
        Whether to compute the affinity matrix in a sparse format. Default is "auto".
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Union[str, FaissConfig] = None,
        verbose: bool = False,
        compile: bool = False,
        sparsity: bool = True,
        random_state: float = None,
        _pre_processed: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            compile=compile,
            _pre_processed=_pre_processed,
        )
        self.sparsity = sparsity

    @property
    def sparsity(self):
        """Return the sparsity of the affinity matrix."""
        return self._sparsity

    @sparsity.setter
    def sparsity(self, value):
        """Set the sparsity of the affinity matrix."""
        self._sparsity = bool_arg(value)

    def __call__(
        self, X: Union[torch.Tensor, np.ndarray], return_indices: bool = True, **kwargs
    ):
        r"""Compute the affinity matrix from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix.
        indices : torch.Tensor
            If return_indices is True, returns the indices of the non-zero elements
            in the affinity matrix if sparsity is enabled. Otherwise, returns None.
        """
        if not self._pre_processed:
            X = to_torch(X)
        return self._compute_sparse_affinity(X, return_indices, **kwargs)

    def _compute_sparse_affinity(
        self, X: torch.Tensor, return_indices: bool = True, **kwargs
    ):
        r"""Compute the sparse affinity matrix from the input data.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.
        return_indices : bool, optional
            If True, returns the indices of the non-zero elements in the affinity matrix
            if sparsity is enabled. Default is False.

        Raises
        ------
        NotImplementedError
            If the `_compute_sparse_affinity` method is not implemented by
            the subclass, a NotImplementedError is raised.
        """
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_sparse_affinity` method is not implemented."
        )


class SparseLogAffinity(SparseAffinity, LogAffinity):
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
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Which backend to use for handling sparsity and memory efficiency.
        Can be:
        - "keops": Use KeOps for memory-efficient symbolic computations
        - "faiss": Use FAISS for fast k-NN computations with default settings
        - None: Use standard PyTorch operations
        - FaissConfig object: Use FAISS with custom configuration
        Default is None.
    verbose : bool, optional
        If True, prints additional information during computation. Default is False.
    compile : bool, optional
        Whether to compile the affinity matrix computation. Default is False.
    sparsity : bool or 'auto', optional
        Whether to compute the affinity matrix in a sparse format. Default is "auto".
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """

    def __call__(
        self,
        X: Union[torch.Tensor, np.ndarray],
        log: bool = False,
        return_indices: bool = True,
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
        if not self._pre_processed:
            X = to_torch(X)

        if return_indices:
            log_affinity, indices = self._compute_sparse_log_affinity(
                X, return_indices, **kwargs
            )
            affinity_to_return = log_affinity if log else log_affinity.exp()
            return (affinity_to_return, indices)
        else:
            log_affinity = self._compute_sparse_log_affinity(
                X, return_indices, **kwargs
            )
            affinity_to_return = log_affinity if log else log_affinity.exp()
            return affinity_to_return

    def _compute_sparse_log_affinity(
        self, X: torch.Tensor, return_indices: bool = False, **kwargs
    ):
        r"""Compute the log affinity matrix in a sparse format from the input data.

        This method must be overridden by subclasses.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.
        return_indices : bool, optional
            If True, returns the indices of the non-zero elements in the affinity matrix
            if sparsity is enabled. Default is False.

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
