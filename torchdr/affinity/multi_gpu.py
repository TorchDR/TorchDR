"""Multi-GPU affinity computation classes.

This module provides base functionality and implementations for distributing
affinity computations across multiple GPUs. Each GPU processes its own chunk
of data independently.
"""

import torch
from typing import Optional, Union, List

from torchdr.affinity.base import SparseAffinity, SparseLogAffinity
from torchdr.affinity.entropic import EntropicAffinity
from torchdr.affinity.knn_normalized import UMAPAffinity
from torchdr.distance.faiss import FaissConfig, pairwise_distances_faiss
from torchdr.utils import check_neighbor_param
from torchdr.utils.operators import logsumexp_red


class MultiGPUAffinityMixin:
    """Mixin for multi-GPU affinity computation.

    Each GPU processes its own chunk of the affinity matrix independently.
    """

    def __init__(self, devices: Optional[Union[int, List[int]]] = None):
        """Initialize multi-GPU affinity.

        Parameters
        ----------
        devices : Optional[Union[int, List[int]]], default=None
            GPU device(s) to use. If None, uses all available GPUs.
            If int, uses that single GPU. If list, uses specified GPUs.
        """
        if devices is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise RuntimeError("No GPUs available for multi-GPU affinity")
            self.devices = list(range(n_gpus))
        elif isinstance(devices, int):
            self.devices = [devices]
        else:
            self.devices = devices

        self.n_devices = len(self.devices)
        self.is_multi_gpu = self.n_devices > 1

    def get_chunk_range(self, n_samples: int, device_id: int) -> tuple:
        """Get the range of samples for a specific GPU.

        Parameters
        ----------
        n_samples : int
            Total number of samples.
        device_id : int
            GPU device ID (index in self.devices list).

        Returns
        -------
        start : int
            Start index for this GPU's chunk.
        end : int
            End index for this GPU's chunk.
        """
        chunk_size = n_samples // self.n_devices
        remainder = n_samples % self.n_devices

        # First 'remainder' GPUs get chunk_size + 1 samples
        if device_id < remainder:
            start = device_id * (chunk_size + 1)
            end = start + chunk_size + 1
        else:
            start = device_id * chunk_size + remainder
            end = start + chunk_size

        return start, end

    def get_current_device_id(self) -> int:
        """Get the index of the current GPU in self.devices list.

        Returns
        -------
        device_id : int
            Index in self.devices list corresponding to current CUDA device.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        current_device = torch.cuda.current_device()
        if current_device not in self.devices:
            raise RuntimeError(
                f"Current CUDA device {current_device} not in configured devices {self.devices}"
            )

        return self.devices.index(current_device)


class MultiGPUSparseAffinity(MultiGPUAffinityMixin, SparseAffinity):
    """Base class for multi-GPU sparse affinity matrices.

    Each GPU computes and stores only its chunk of the affinity matrix.
    """

    def __init__(self, devices: Optional[Union[int, List[int]]] = None, **kwargs):
        """Initialize multi-GPU sparse affinity.

        Parameters
        ----------
        devices : Optional[Union[int, List[int]]], default=None
            GPU device(s) to use.
        **kwargs
            Additional arguments passed to SparseAffinity.
        """
        MultiGPUAffinityMixin.__init__(self, devices)
        SparseAffinity.__init__(self, **kwargs)


class MultiGPUSparseLogAffinity(MultiGPUAffinityMixin, SparseLogAffinity):
    """Base class for multi-GPU sparse log-affinity matrices.

    Each GPU computes and stores only its chunk of the log-affinity matrix.
    """

    def __init__(self, devices: Optional[Union[int, List[int]]] = None, **kwargs):
        """Initialize multi-GPU sparse log-affinity.

        Parameters
        ----------
        devices : Optional[Union[int, List[int]]], default=None
            GPU device(s) to use.
        **kwargs
            Additional arguments passed to SparseLogAffinity.
        """
        MultiGPUAffinityMixin.__init__(self, devices)
        SparseLogAffinity.__init__(self, **kwargs)


class EntropicAffinityMultiGPU(EntropicAffinity, MultiGPUSparseLogAffinity):
    """Multi-GPU Entropic Affinity.

    Each GPU computes and stores only its chunk of the affinity matrix.
    The kNN computation uses FAISS multi-GPU, then each GPU processes
    only its assigned chunk of samples.
    """

    def __init__(
        self,
        perplexity: float = 30.0,
        devices: Optional[Union[int, List[int]]] = None,
        max_iter: int = 1000,
        sparsity: bool = True,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        verbose: bool = False,
        compile: bool = False,
    ):
        """Initialize multi-GPU entropic affinity.

        Parameters
        ----------
        perplexity : float, default=30.0
            Target perplexity for the affinity matrix.
        devices : Optional[Union[int, List[int]]], default=None
            GPU device(s) to use. If None, uses all available GPUs.
            If int, uses that single GPU. If list, uses specified GPUs.
        max_iter : int, default=1000
            Maximum number of iterations for the binary search.
        sparsity : bool, default=True
            Whether to use sparse affinity matrix (with k nearest neighbors).
        metric : str, default="sqeuclidean"
            Distance metric to use. Options: "sqeuclidean", "euclidean", "angular".
        zero_diag : bool, default=True
            Whether to set the diagonal of the affinity matrix to zero.
        verbose : bool, default=False
            Whether to print progress messages.
        compile : bool, default=False
            Whether to use torch.compile for optimization (requires PyTorch 2.0+).

        Notes
        -----
        Multi-GPU mode requires backend='faiss' and is automatically set.
        The 'device' parameter from EntropicAffinity is replaced by 'devices'.
        """
        MultiGPUSparseLogAffinity.__init__(self, devices=devices)

        # Multi-GPU only works with FAISS backend
        EntropicAffinity.__init__(
            self,
            perplexity=perplexity,
            max_iter=max_iter,
            sparsity=sparsity,
            metric=metric,
            zero_diag=zero_diag,
            device="cuda",  # Will be overridden by multi-GPU logic
            backend="faiss",  # Force FAISS backend
            verbose=verbose,
            compile=compile,
        )

    def _distance_matrix(
        self, X: torch.Tensor, k: int = None, return_indices: bool = False
    ):
        """Compute distance matrix using FAISS with output on CPU.

        Keeps the full kNN results on CPU to save GPU memory.
        Each GPU will then load only its chunk.
        """
        config = FaissConfig(
            device=self.devices if self.is_multi_gpu else self.devices[0],
            output_device="cpu",  # Keep output on CPU for memory efficiency
        )

        C_cpu, indices_cpu = pairwise_distances_faiss(
            X, k=k, metric=self.metric, config=config
        )

        if return_indices:
            return C_cpu, indices_cpu
        return C_cpu

    def _compute_sparse_log_affinity(
        self, X: torch.Tensor, return_indices: bool = True, **kwargs
    ):
        """Compute only the current GPU's chunk of the affinity matrix.

        Each GPU processes its assigned chunk independently using the shared
        affinity computation logic from the parent class.
        """
        n_samples = X.shape[0]
        device_id = self.get_current_device_id() if self.is_multi_gpu else 0
        start, end = (
            self.get_chunk_range(n_samples, device_id)
            if self.is_multi_gpu
            else (0, n_samples)
        )

        n_samples_tensor = torch.tensor(n_samples, dtype=X.dtype, device=X.device)
        perplexity = check_neighbor_param(self.perplexity, n_samples_tensor)
        k = min(3 * perplexity, n_samples - 1)
        k = check_neighbor_param(k, n_samples_tensor)

        C_cpu, indices_cpu = self._distance_matrix(X, k=k, return_indices=True)

        current_device = torch.device(f"cuda:{self.devices[device_id]}")
        C_chunk = C_cpu[start:end].to(current_device)
        indices_chunk = indices_cpu[start:end].to(current_device)

        log_affinity_chunk, eps_chunk = self._compute_log_affinity_from_cost(
            C_chunk, n_samples
        )

        self.register_buffer("eps_", eps_chunk, persistent=False)

        log_normalization = logsumexp_red(log_affinity_chunk, dim=1)
        self.register_buffer("log_normalization_", log_normalization, persistent=False)
        log_affinity_chunk -= self.log_normalization_

        log_affinity_chunk -= torch.log(
            torch.tensor(n_samples, dtype=C_chunk.dtype, device=current_device)
        )

        self.chunk_start_ = start
        self.chunk_end_ = end
        self.chunk_size_ = end - start

        if return_indices:
            return log_affinity_chunk, indices_chunk
        return log_affinity_chunk


class UMAPAffinityMultiGPU(UMAPAffinity, MultiGPUSparseAffinity):
    """Multi-GPU UMAP Affinity.

    Each GPU computes and stores only its chunk of the affinity matrix.
    The kNN computation uses FAISS multi-GPU, then each GPU processes
    only its assigned chunk of samples.
    """

    def __init__(
        self,
        n_neighbors: float = 30.0,
        devices: Optional[Union[int, List[int]]] = None,
        max_iter: int = 1000,
        sparsity: bool = True,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        verbose: bool = False,
        compile: bool = False,
        symmetrize: bool = True,
    ):
        """Initialize multi-GPU UMAP affinity.

        Parameters
        ----------
        n_neighbors : float, default=30.0
            Number of effective nearest neighbors to consider. Similar to perplexity.
        devices : Optional[Union[int, List[int]]], default=None
            GPU device(s) to use. If None, uses all available GPUs.
            If int, uses that single GPU. If list, uses specified GPUs.
        max_iter : int, default=1000
            Maximum number of iterations for the root search.
        sparsity : bool, default=True
            Whether to use sparse affinity matrix (with k nearest neighbors).
        metric : str, default="sqeuclidean"
            Distance metric to use. Options: "sqeuclidean", "euclidean", "angular".
        zero_diag : bool, default=True
            Whether to set the diagonal of the affinity matrix to zero.
        verbose : bool, default=False
            Whether to print progress messages.
        compile : bool, default=False
            Whether to use torch.compile for optimization (requires PyTorch 2.0+).
        symmetrize : bool, default=True
            Whether to symmetrize the affinity matrix.

        Notes
        -----
        Multi-GPU mode requires backend='faiss' and is automatically set.
        The 'device' parameter from UMAPAffinity is replaced by 'devices'.
        """
        MultiGPUSparseAffinity.__init__(self, devices=devices)

        # Multi-GPU only works with FAISS backend
        UMAPAffinity.__init__(
            self,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            sparsity=sparsity,
            metric=metric,
            zero_diag=zero_diag,
            device="cuda",  # Will be overridden by multi-GPU logic
            backend="faiss",  # Force FAISS backend
            verbose=verbose,
            compile=compile,
            symmetrize=symmetrize,
        )

    def _distance_matrix(
        self, X: torch.Tensor, k: int = None, return_indices: bool = False
    ):
        """Compute distance matrix using FAISS with output on CPU.

        Keeps the full kNN results on CPU to save GPU memory.
        Each GPU will then load only its chunk.
        """
        config = FaissConfig(
            device=self.devices if self.is_multi_gpu else self.devices[0],
            output_device="cpu",  # Keep output on CPU for memory efficiency
        )

        C_cpu, indices_cpu = pairwise_distances_faiss(
            X, k=k, metric=self.metric, config=config
        )

        if return_indices:
            return C_cpu, indices_cpu
        return C_cpu

    def _compute_sparse_affinity(
        self, X: torch.Tensor, return_indices: bool = True, **kwargs
    ):
        """Compute only the current GPU's chunk of the affinity matrix.

        Each GPU processes its assigned chunk independently using the shared
        affinity computation logic from the parent class.
        """
        n_samples = X.shape[0]
        device_id = self.get_current_device_id() if self.is_multi_gpu else 0
        start, end = (
            self.get_chunk_range(n_samples, device_id)
            if self.is_multi_gpu
            else (0, n_samples)
        )

        n_samples_tensor = torch.tensor(n_samples, dtype=X.dtype, device=X.device)
        n_neighbors = check_neighbor_param(self.n_neighbors, n_samples_tensor)

        C_cpu, indices_cpu = self._distance_matrix(
            X, k=n_neighbors, return_indices=True
        )

        current_device = torch.device(f"cuda:{self.devices[device_id]}")
        C_chunk = C_cpu[start:end].to(current_device)
        indices_chunk = indices_cpu[start:end].to(current_device)

        affinity_chunk, rho_chunk, eps_chunk = self._compute_affinity_from_cost(
            C_chunk, n_samples
        )

        self.register_buffer("rho_", rho_chunk, persistent=False)
        self.register_buffer("eps_", eps_chunk, persistent=False)

        # Note: In multi-GPU mode, symmetrization would require communication
        # between GPUs. For now, we handle non-symmetric chunks.
        if self.symmetrize and self.is_multi_gpu:
            self.logger.warning(
                "Full symmetrization not supported in multi-GPU mode. "
                "Using asymmetric affinities for efficiency."
            )

        self.chunk_start_ = start
        self.chunk_end_ = end
        self.chunk_size_ = end - start

        if return_indices:
            return affinity_chunk, indices_chunk
        return affinity_chunk
