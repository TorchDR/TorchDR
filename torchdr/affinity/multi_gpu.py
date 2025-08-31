"""Multi-GPU affinity computation classes.

This module provides base functionality and implementations for distributing
affinity computations across multiple GPUs. Each GPU processes its own chunk
of data independently.
"""

import torch
import torch.distributed as dist

from torchdr.affinity.entropic import EntropicAffinity
from torchdr.affinity.knn_normalized import UMAPAffinity
from torchdr.distance.faiss import pairwise_distances_faiss, FaissConfig


class MultiGPUAffinityMixin:
    """Mixin for multi-GPU affinity computation.

    Each GPU processes its own chunk of the affinity matrix independently.
    """

    def __init__(self, zero_diag: bool = True):
        """Initialize multi-GPU affinity for distributed mode only.

        Must be launched with torchrun for distributed execution.
        """
        # Require distributed mode
        if not dist.is_initialized():
            raise RuntimeError(
                "[TorchDR] MultiGPUAffinityMixin requires distributed mode. "
                "Please launch with torchrun, e.g.: "
                "torchrun --nproc_per_node=4 your_script.py"
            )

        # In distributed mode, each process handles one GPU
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.n_devices = self.world_size
        self.is_multi_gpu = self.world_size > 1
        self.zero_diag = zero_diag

    def _compute_chunk_info(self, n_samples: int):
        """Compute and store chunk boundaries for this rank.

        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.
        """
        chunk_size = n_samples // self.n_devices
        remainder = n_samples % self.n_devices

        # First 'remainder' ranks get chunk_size + 1 samples
        if self.rank < remainder:
            self.chunk_start_ = self.rank * (chunk_size + 1)
            self.chunk_end_ = self.chunk_start_ + chunk_size + 1
        else:
            self.chunk_start_ = self.rank * chunk_size + remainder
            self.chunk_end_ = self.chunk_start_ + chunk_size

        self.chunk_size_ = self.chunk_end_ - self.chunk_start_

    def _distance_matrix(
        self, X: torch.Tensor, k: int = None, return_indices: bool = False
    ):
        """Compute distances for this GPU's chunk of points.

        Each GPU computes k-NN distances where:
        - Database (keys): Full dataset X
        - Queries: This GPU's assigned chunk of X

        This way each GPU computes and stores only its rows of the affinity matrix.

        Parameters
        ----------
        X : torch.Tensor
            Full input data tensor.
        k : int
            Number of nearest neighbors. Must be specified.
        return_indices : bool, default=False
            Whether to return indices along with distances.

        Returns
        -------
        distances : torch.Tensor
            Distance matrix for this GPU's chunk. Shape (chunk_size, k).
        indices : torch.Tensor, optional
            Indices of nearest neighbors if return_indices=True.
        """
        if k is None:
            raise ValueError(
                "[TorchDR] Multi-GPU affinity computation requires k to be specified (sparse mode only). "
                "Full dense affinity matrices are not supported due to memory constraints."
            )

        n_samples = X.shape[0]

        # Compute and store chunk info for this rank
        self._compute_chunk_info(n_samples)

        # Get this GPU's query chunk
        X_chunk = X[self.chunk_start_ : self.chunk_end_]

        # In distributed mode with torchrun, each process uses its current device
        gpu_device = torch.cuda.current_device()

        faiss_config = FaissConfig(
            device=gpu_device,
            use_float16=False,  # Use float32 for better precision
        )

        # Compute k-NN where:
        # - X_chunk (queries): this GPU's chunk of points
        # - X (database): full dataset
        # This gives distances from chunk points to all points in dataset
        #
        # Since X_chunk is a slice of X, we need to handle diagonal exclusion manually
        k_search = k + 1 if self.zero_diag else k

        distances, indices = pairwise_distances_faiss(
            X_chunk,
            k=k_search,
            Y=X,  # Full dataset as database
            metric=self.metric,
            config=faiss_config,
            exclude_diag=False,  # Can't use this since X_chunk != X
        )

        # Remove the closest neighbor (should be self with distance ~0)
        # This works because X_chunk is a subset of X, so each point finds itself
        if self.zero_diag:
            # Discard first column (closest neighbor, which should be self)
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        if self.verbose:
            self.logger.info(
                f"Rank {self.rank}: Computed distances for chunk [{self.chunk_start_}:{self.chunk_end_}] "
                f"(shape: {distances.shape})"
            )

        if return_indices:
            return distances, indices
        return distances


class EntropicAffinityMultiGPU(MultiGPUAffinityMixin, EntropicAffinity):
    """Multi-GPU Entropic Affinity for distributed training.

    Each GPU computes and stores only its chunk of the affinity matrix.
    Must be launched with torchrun for distributed execution.
    """

    def __init__(
        self,
        perplexity: float = 30.0,
        max_iter: int = 1000,
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
        max_iter : int, default=1000
            Maximum number of iterations for the binary search.
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
        Requires distributed mode via torchrun.
        Always uses sparsity=True and backend='faiss'.
        """
        MultiGPUAffinityMixin.__init__(self, zero_diag=zero_diag)

        EntropicAffinity.__init__(
            self,
            perplexity=perplexity,
            max_iter=max_iter,
            sparsity=True,  # Always use sparsity in multi-GPU mode
            metric=metric,
            zero_diag=zero_diag,
            device="cuda",  # Each process uses its assigned GPU
            backend="faiss",  # Force FAISS backend
            verbose=verbose,
            compile=compile,
        )


class UMAPAffinityMultiGPU(MultiGPUAffinityMixin, UMAPAffinity):
    """Multi-GPU UMAP Affinity for distributed training.

    Each GPU computes and stores only its chunk of the affinity matrix.
    Must be launched with torchrun for distributed execution.
    """

    def __init__(
        self,
        n_neighbors: float = 30.0,
        max_iter: int = 1000,
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
        max_iter : int, default=1000
            Maximum number of iterations for the root search.
        metric : str, default="sqeuclidean"
            Distance metric to use. Options: "sqeuclidean", "euclidean", "angular".
        zero_diag : bool, default=True
            Whether to set the diagonal of the affinity matrix to zero.
        verbose : bool, default=False
            Whether to print progress messages.
        compile : bool, default=False
            Whether to use torch.compile for optimization (requires PyTorch 2.0+).
        symmetrize : bool, default=True
            Whether to symmetrize the affinity matrix (limited in multi-GPU mode).

        Notes
        -----
        Requires distributed mode via torchrun.
        Always uses sparsity=True and backend='faiss'.
        """
        MultiGPUAffinityMixin.__init__(self, zero_diag=zero_diag)

        # Warn if symmetrization was requested in multi-GPU mode
        if symmetrize and self.is_multi_gpu:
            if verbose:
                print(
                    "[TorchDR] WARNING: Symmetrization not supported in multi-GPU mode. "
                    "Setting symmetrize=False. Use single-GPU mode if symmetrization is required."
                )
            symmetrize = False

        UMAPAffinity.__init__(
            self,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            sparsity=True,  # Always sparse in multi-GPU mode
            metric=metric,
            zero_diag=zero_diag,
            device="cuda",  # Each process uses its assigned GPU
            backend="faiss",  # Force FAISS backend
            verbose=verbose,
            compile=compile,
            symmetrize=symmetrize,  # Will be False in multi-GPU mode
        )
