"""Base classes for affinity matrices."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC
from typing import Union, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchdr.utils import (
    to_torch,
    bool_arg,
    set_logger,
    DistributedContext,
)

from torchdr.distance import (
    pairwise_distances,
    FaissConfig,
)

import torch.distributed as dist


class Affinity(nn.Module, ABC):
    r"""Base class for affinity matrices.

    Parameters
    ----------
    metric : str, optional
        Distance metric for pairwise distances. Default is "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal to zero. Default is True.
    device : str, optional
        Device for computation. ``"auto"`` uses the input data's device.
        Default is "auto".
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Backend for handling sparsity and memory efficiency.
        Default is None (standard PyTorch).
    verbose : bool, optional
        Verbosity. Default is False.
    compile : bool, optional
        Whether to compile the affinity computation. Default is False.
    _pre_processed : bool, optional
        If True, skips ``to_torch`` conversion (inputs are already tensors
        on the correct device). Default is False.
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
        self.device = device if device is not None else "auto"
        self.backend = backend
        self.verbose = bool_arg(verbose)
        self.random_state = random_state
        self.compile = compile
        self._pre_processed = _pre_processed

        self.logger = set_logger(self.__class__.__name__, self.verbose)

    # --- Public API ---

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

    # --- Core computation (must be implemented by subclasses) ---

    def _compute_affinity(self, X: torch.Tensor):
        r"""Compute the affinity matrix. Must be overridden by subclasses."""
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_affinity` method is not implemented."
        )

    # --- Distance computation ---

    def _distance_matrix(
        self, X: torch.Tensor, k: int = None, return_indices: bool = False
    ):
        r"""Compute the pairwise distance matrix.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data.
        k : int, optional
            Number of nearest neighbors. Default is None (full matrix).
        return_indices : bool, optional
            Whether to return k-NN indices. Default is False.

        Returns
        -------
        C : torch.Tensor or pykeops.torch.LazyTensor
            The pairwise distance matrix.
        """
        return pairwise_distances(
            X=X,
            metric=self.metric,
            backend=self.backend,
            exclude_diag=self.zero_diag,
            k=k,
            return_indices=return_indices,
            device=self.device,
        )

    # --- Utilities ---

    def _get_compute_device(self, X):
        """Return the target device (from ``self.device`` or inferred from X)."""
        if self.device != "auto":
            return self.device

        if isinstance(X, DataLoader):
            from torchdr.distance.faiss import get_dataloader_metadata

            metadata = get_dataloader_metadata(X)
            if metadata is not None and "device" in metadata:
                return metadata["device"]
            for batch in X:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                return batch.device
            return torch.device("cpu")

        return X.device

    def _get_n_samples(self, X):
        """Return the number of samples in the input."""
        if isinstance(X, DataLoader):
            return len(X.dataset)
        return X.shape[0]

    def _get_dtype(self, X):
        """Return the dtype of the input."""
        if isinstance(X, DataLoader):
            from torchdr.distance.faiss import get_dataloader_metadata

            metadata = get_dataloader_metadata(X)
            if metadata is not None:
                return metadata["dtype"]
            for batch in X:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                return batch.dtype
            raise ValueError("[TorchDR] DataLoader is empty, cannot determine dtype.")
        return X.dtype

    # --- Memory management ---

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

    Subclasses must implement :meth:`_compute_log_affinity`.

    Parameters
    ----------
    metric : str, optional
        Distance metric for pairwise distances. Default is "sqeuclidean".
    device : str, optional
        Device for computation. ``"auto"`` uses the input data's device.
        Default is "auto".
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Backend for handling sparsity and memory efficiency.
        Default is None (standard PyTorch).
    verbose : bool, optional
        Verbosity. Default is False.
    compile : bool, optional
        Whether to compile the affinity computation. Default is False.
    _pre_processed : bool, optional
        If True, skips ``to_torch`` conversion. Default is False.
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
        self,
        X: Union[torch.Tensor, np.ndarray],
        log: bool = False,
        **kwargs: Any,
    ):
        r"""Compute the affinity matrix (or its log) from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.
        log : bool, optional
            If True, returns the log affinity. Otherwise, exponentiates it.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The affinity matrix (or log affinity if ``log=True``).
        """
        if not self._pre_processed:
            X = to_torch(X)
        log_affinity = self._compute_log_affinity(X, **kwargs)
        if log:
            return log_affinity
        else:
            return log_affinity.exp()

    def _compute_log_affinity(self, X: torch.Tensor, **kwargs):
        r"""Compute the log affinity matrix. Must be overridden by subclasses."""
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_log_affinity` method is not implemented."
        )


class SparseAffinity(Affinity):
    r"""Base class for sparse affinity matrices.

    Returns the affinity matrix in rectangular format (n_samples, k) with
    the corresponding k-NN indices when sparsity is enabled. Otherwise,
    returns the full (n_samples, n_samples) matrix.

    **Distributed training:** When ``distributed='auto'`` (default) and
    launched with ``torchrun``, each GPU processes a chunk of the dataset
    in parallel. Requires ``sparsity=True`` and ``backend="faiss"``.

    Subclasses must implement :meth:`_compute_sparse_affinity`.

    Parameters
    ----------
    metric : str, optional
        Distance metric for pairwise distances. Default is "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal to zero. Default is True.
    device : str, optional
        Device for computation. ``"auto"`` uses the input data's device.
        Default is "auto".
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Backend for handling sparsity and memory efficiency.
        Default is None (standard PyTorch).
    verbose : bool, optional
        Verbosity. Default is False.
    compile : bool, optional
        Whether to compile the affinity computation. Default is False.
    sparsity : bool or 'auto', optional
        Whether to use sparse (rectangular) format. Default is True.
    distributed : bool or 'auto', optional
        Whether to use distributed multi-GPU computation.
        ``"auto"`` detects ``torchrun`` automatically. Default is "auto".
    _pre_processed : bool, optional
        If True, skips ``to_torch`` conversion. Default is False.
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
        distributed: Union[bool, str] = "auto",
        random_state: float = None,
        _pre_processed: bool = False,
    ):
        # --- Distributed setup ---
        if distributed == "auto":
            self.distributed = dist.is_initialized()
        else:
            self.distributed = bool(distributed)

        if self.distributed:
            if not dist.is_initialized():
                raise RuntimeError(
                    "[TorchDR] distributed=True requires launching with "
                    "torchrun. "
                    "Example: torchrun --nproc_per_node=4 your_script.py"
                )

            self.dist_ctx = DistributedContext()
            self.rank = self.dist_ctx.rank
            self.world_size = self.dist_ctx.world_size
            self.is_multi_gpu = self.world_size > 1

            if device == "cpu":
                raise ValueError(
                    "[TorchDR] Distributed mode requires GPU (device cannot be 'cpu')"
                )
            device = torch.device(f"cuda:{self.dist_ctx.local_rank}")

            # Force sparsity and FAISS backend for distributed mode
            self._sparsity_forced = not sparsity
            if self._sparsity_forced:
                sparsity = True

            self._backend_forced = backend not in [
                "faiss",
                None,
            ] and not isinstance(backend, FaissConfig)
            if self._backend_forced:
                self._original_backend = backend
                backend = "faiss"
        else:
            self.dist_ctx = None
            self.rank = 0
            self.world_size = 1
            self.is_multi_gpu = False

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

        if self.distributed and self.verbose:
            if self._sparsity_forced:
                self.logger.warning(
                    "Distributed mode requires sparsity=True, enabling sparsity."
                )
            if self._backend_forced:
                self.logger.warning(
                    f"Distributed mode requires FAISS backend, "
                    f"switching from '{self._original_backend}' to 'faiss'."
                )
            if self.is_multi_gpu:
                self.logger.info(
                    f"Distributed mode enabled: rank {self.rank}/{self.world_size}"
                )

    # --- Sparsity property ---

    @property
    def sparsity(self):
        """Return the sparsity setting."""
        return self._sparsity

    @sparsity.setter
    def sparsity(self, value):
        """Set the sparsity setting."""
        self._sparsity = bool_arg(value)

    # --- Public API ---

    def __call__(
        self,
        X: Union[torch.Tensor, np.ndarray],
        return_indices: bool = True,
        **kwargs,
    ):
        r"""Compute the sparse affinity matrix from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.
        return_indices : bool, optional
            Whether to return k-NN indices. Default is True.

        Returns
        -------
        affinity_matrix : torch.Tensor
            The computed affinity matrix.
        indices : torch.Tensor or None
            k-NN indices if ``return_indices=True`` and sparsity is enabled.
        """
        if not self._pre_processed:
            X = to_torch(X)
        return self._compute_sparse_affinity(X, return_indices, **kwargs)

    # --- Core computation (must be implemented by subclasses) ---

    def _compute_sparse_affinity(
        self, X: torch.Tensor, return_indices: bool = True, **kwargs
    ):
        r"""Compute the sparse affinity matrix. Must be overridden."""
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_sparse_affinity` method is not implemented."
        )

    # --- Distance computation ---

    def _distance_matrix(
        self, X: torch.Tensor, k: int = None, return_indices: bool = False
    ):
        """Compute pairwise distances, passing distributed context if active.

        Parameters
        ----------
        X : torch.Tensor
            Input data.
        k : int, optional
            Number of nearest neighbors.
        return_indices : bool, default=False
            Whether to return k-NN indices.

        Returns
        -------
        distances : torch.Tensor
            Distance matrix.
        indices : torch.Tensor, optional
            Indices if ``return_indices=True``.
        """
        result = pairwise_distances(
            X=X,
            metric=self.metric,
            backend=self.backend,
            exclude_diag=self.zero_diag,
            k=k,
            return_indices=return_indices,
            device=self.device,
            distributed_ctx=self.dist_ctx if self.distributed else None,
        )

        # Store chunk bounds for downstream use (e.g. distributed symmetrization)
        if self.distributed and self.dist_ctx is not None:
            chunk_start, chunk_end = self.dist_ctx.compute_chunk_bounds(
                self._get_n_samples(X)
            )
            self.chunk_start_ = chunk_start
            self.chunk_end_ = chunk_end
            self.chunk_size_ = chunk_end - chunk_start

        return result


class SparseLogAffinity(SparseAffinity, LogAffinity):
    r"""Base class for sparse log affinity matrices.

    Combines :class:`SparseAffinity` (sparse format, distributed support)
    with :class:`LogAffinity` (log-domain computation).

    Subclasses must implement :meth:`_compute_sparse_log_affinity`.

    Parameters
    ----------
    metric : str, optional
        Distance metric for pairwise distances. Default is "sqeuclidean".
    zero_diag : bool, optional
        Whether to set the diagonal to zero. Default is True.
    device : str, optional
        Device for computation. ``"auto"`` uses the input data's device.
        Default is "auto".
    backend : {"keops", "faiss", None} or FaissConfig, optional
        Backend for handling sparsity and memory efficiency.
        Default is None (standard PyTorch).
    verbose : bool, optional
        Verbosity. Default is False.
    compile : bool, optional
        Whether to compile the affinity computation. Default is False.
    sparsity : bool or 'auto', optional
        Whether to use sparse (rectangular) format. Default is True.
    distributed : bool or 'auto', optional
        Whether to use distributed multi-GPU computation.
        ``"auto"`` detects ``torchrun`` automatically. Default is "auto".
    _pre_processed : bool, optional
        If True, skips ``to_torch`` conversion. Default is False.
    """

    def __call__(
        self,
        X: Union[torch.Tensor, np.ndarray],
        log: bool = False,
        return_indices: bool = True,
        **kwargs,
    ):
        r"""Compute the sparse (log) affinity matrix from the input data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.
        log : bool, optional
            If True, returns the log affinity. Otherwise, exponentiates it.
        return_indices : bool, optional
            Whether to return k-NN indices. Default is True.

        Returns
        -------
        affinity_matrix : torch.Tensor
            The affinity matrix (or log affinity if ``log=True``).
        indices : torch.Tensor or None
            k-NN indices if ``return_indices=True`` and sparsity is enabled.
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
        r"""Compute the sparse log affinity matrix. Must be overridden."""
        raise NotImplementedError(
            "[TorchDR] ERROR : `_compute_sparse_log_affinity` method is "
            "not implemented."
        )
