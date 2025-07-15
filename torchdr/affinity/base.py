"""Base classes for affinity matrices."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC
from typing import Union, Any

import numpy as np
import torch

from torchdr.utils import (
    LazyTensorType,
    handle_keops,
    to_torch,
    bool_arg,
    set_logger,
)

from torchdr.distance import (
    pairwise_distances,
    symmetric_pairwise_distances_indices,
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
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
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
        backend: str = None,
        verbose: bool = False,
        random_state: float = None,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
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

    @handle_keops
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
        # The `@handle_keops` decorator sets `self.backend_`, which is used below.
        return pairwise_distances(
            X=X,
            metric=self.metric,
            backend=self.backend_,
            exclude_diag=self.zero_diag,  # infinite distance means zero affinity
            k=k,
            return_indices=return_indices,
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
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
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
        backend: str = None,
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
            X = to_torch(X, device=self.device)
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
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
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
        backend: str = None,
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
            X = to_torch(X, device=self.device)
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
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
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
            X = to_torch(X, device=self.device)

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
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
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
        backend: str = None,
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
        self._pre_processed = _pre_processed

    def __call__(
        self,
        X: Union[torch.Tensor, np.ndarray],
        Y: Union[torch.Tensor, np.ndarray] = None,
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
        if not self._pre_processed:
            X = to_torch(X, device=self.device)
            if Y is not None:
                Y = to_torch(Y, device=self.device)
        C = self._distance_matrix(X=X, Y=Y, indices=indices, **kwargs)
        return self._affinity_formula(C)

    def _affinity_formula(self, C: Union[torch.Tensor, LazyTensorType]):
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

    @handle_keops
    def _distance_matrix(
        self,
        X: Union[torch.Tensor, np.ndarray],
        Y: Union[torch.Tensor, np.ndarray] = None,
        indices: torch.Tensor = None,
    ):
        r"""Compute the pairwise distance matrix from the input data.

        It uses the specified metric and optionally leveraging KeOps
        for memory efficient computation.
        It supports computing the full pairwise distance matrix, the pairwise
        distance matrix for a given set of indices, and the cross-distance
        matrix between two datasets.

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
            value of the `backend` attribute. If `backend` is `keops`, a KeOps LazyTensor
            is returned. Otherwise, a torch.Tensor is returned.
        """
        if Y is not None and indices is not None:
            raise NotImplementedError(
                "[TorchDR] ERROR : transform method cannot be called with both Y "
                "and indices at the same time."
            )

        # Note: The `backend_` attribute is set by the `@handle_keops` decorator.

        elif Y is not None:  # Case 1: Cross-distance matrix
            return pairwise_distances(X, Y, metric=self.metric, backend=self.backend_)

        elif indices is not None:  # Case 2: Sparse self-distance matrix
            return symmetric_pairwise_distances_indices(
                X, indices=indices, metric=self.metric
            )

        else:  # Case 3: Full self-distance matrix (with or without the diagonal)
            return pairwise_distances(
                X,
                metric=self.metric,
                backend=self.backend_,
                exclude_diag=self.zero_diag,  # infinite distance is zero affinity
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
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
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
        backend: str = None,
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
        self,
        X: Union[torch.Tensor, np.ndarray],
        Y: Union[torch.Tensor, np.ndarray] = None,
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
        if not self._pre_processed:
            X = to_torch(X, device=self.device)
            if Y is not None:
                Y = to_torch(Y, device=self.device)
        C = self._distance_matrix(X=X, Y=Y, indices=indices, **kwargs)
        log_affinity = self._log_affinity_formula(C)
        if log:
            return log_affinity
        else:
            return log_affinity.exp()

    def _log_affinity_formula(self, C: Union[torch.Tensor, LazyTensorType]):
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
