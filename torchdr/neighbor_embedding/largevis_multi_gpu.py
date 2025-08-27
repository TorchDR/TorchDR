"""Multi-GPU implementation of LargeVis."""

import torch
import torch.distributed as dist
from typing import Optional, Union, List

from torchdr.neighbor_embedding.largevis import LargeVis
from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from torchdr.affinity.multi_gpu import EntropicAffinityMultiGPU
from torchdr.utils import cross_entropy_loss, sum_red


class LargeVisMultiGPU(LargeVis):
    """Multi-GPU implementation of LargeVis.

    Each GPU processes its chunk of the affinity matrix independently.
    Embeddings are replicated across all GPUs for loss computation.
    """

    def __init__(
        self,
        perplexity: float = 30,
        n_components: int = 2,
        devices: Optional[Union[int, List[int]]] = None,
        lr: Union[float, str] = "auto",
        optimizer: str = "SGD",
        optimizer_kwargs: Union[dict, str] = "auto",
        scheduler: str = "LinearLR",
        scheduler_kwargs: dict = None,
        init: Union[str, torch.Tensor] = "pca",
        init_scaling: float = 1e-4,
        min_grad_norm: float = 1e-7,
        max_iter: int = 1000,
        verbose: bool = False,
        random_state: Optional[float] = None,
        early_exaggeration_coeff: Optional[float] = None,
        early_exaggeration_iter: Optional[int] = None,
        n_negatives: int = 5,
        sparsity: bool = True,
        check_interval: int = 50,
        discard_NNs: bool = True,
        compile: bool = False,
        shard: bool = True,
    ):
        """Initialize multi-GPU LargeVis.

        Parameters
        ----------
        perplexity : float, default=30
            Number of effective nearest neighbors.
        n_components : int, default=2
            Dimension of the embedding space.
        devices : Optional[Union[int, List[int]]], default=None
            GPU device(s) to use. If None, uses all available GPUs.
        lr : float or 'auto', default='auto'
            Learning rate.
        optimizer : str, default="SGD"
            Optimizer name from torch.optim.
        optimizer_kwargs : dict or 'auto', default='auto'
            Additional optimizer arguments.
        scheduler : str, default="LinearLR"
            Scheduler name from torch.optim.lr_scheduler.
        scheduler_kwargs : dict, optional
            Additional scheduler arguments.
        init : str or torch.Tensor, default='pca'
            Initialization method for embeddings.
        init_scaling : float, default=1e-4
            Scaling factor for initialization.
        min_grad_norm : float, default=1e-7
            Minimum gradient norm for convergence.
        max_iter : int, default=1000
            Maximum number of iterations.
        verbose : bool, default=False
            Whether to print progress.
        random_state : float, optional
            Random seed.
        early_exaggeration_coeff : float, optional
            Early exaggeration coefficient.
        early_exaggeration_iter : int, optional
            Number of early exaggeration iterations.
        n_negatives : int, default=5
            Number of negative samples per point.
        sparsity : bool, default=True
            Whether to use sparse affinity.
        check_interval : int, default=50
            Interval for convergence checks.
        discard_NNs : bool, default=True
            Whether to discard nearest neighbors from negative sampling.
        compile : bool, default=False
            Whether to use torch.compile.
        shard : bool, default=True
            Whether to use IndexShards for FAISS multi-GPU.
        """
        # Setup multi-GPU devices
        if devices is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise RuntimeError("No GPUs available for multi-GPU LargeVis")
            self.devices = list(range(n_gpus))
        elif isinstance(devices, int):
            self.devices = [devices]
        else:
            self.devices = devices

        self.n_devices = len(self.devices)
        self.is_multi_gpu = self.n_devices > 1
        self.shard = shard

        # Initialize distributed if multi-GPU
        if self.is_multi_gpu and not dist.is_initialized():
            # Initialize with NCCL backend for GPU communication
            dist.init_process_group(backend="nccl")

        # Create multi-GPU input affinity
        affinity_in = EntropicAffinityMultiGPU(
            perplexity=perplexity,
            devices=self.devices,
            sparsity=sparsity,
            shard=shard,
            verbose=verbose,
        )

        # Output affinity will be computed directly on each GPU
        affinity_out = None

        # Initialize parent LargeVis
        super().__init__(
            perplexity=perplexity,
            n_components=n_components,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            init=init,
            init_scaling=init_scaling,
            min_grad_norm=min_grad_norm,
            max_iter=max_iter,
            device="cuda",  # Will handle per-GPU
            backend="faiss",  # Always use FAISS for multi-GPU
            verbose=verbose,
            random_state=random_state,
            early_exaggeration_coeff=early_exaggeration_coeff,
            early_exaggeration_iter=early_exaggeration_iter,
            n_negatives=n_negatives,
            sparsity=sparsity,
            check_interval=check_interval,
            discard_NNs=discard_NNs,
            compile=compile,
        )

        # Override affinities with multi-GPU versions
        self.affinity_in = affinity_in
        self.affinity_out = affinity_out

    def _compute_attractive_loss(self):
        """Compute attractive loss using chunked affinity matrix.

        Each GPU computes the loss for its chunk of the affinity matrix.
        The embedding is replicated across all GPUs.
        """
        # Get chunk boundaries for this GPU
        if hasattr(self.affinity_in, "chunk_start_"):
            chunk_start = self.affinity_in.chunk_start_
            chunk_size = self.affinity_in.chunk_size_
        elif hasattr(self, "chunk_start_"):
            chunk_start = self.chunk_start_
            chunk_size = self.chunk_size_
        else:
            # Fallback: assume full dataset if no chunking
            chunk_start = 0
            chunk_size = self.n_samples_in_

        # Get the neighbor indices for this chunk
        if hasattr(self.affinity_in, "indices_"):
            indices_chunk = (
                self.affinity_in.indices_
            )  # Shape: (chunk_size, n_neighbors)
        else:
            indices_chunk = self.NN_indices_

        # Extract embeddings for this chunk and its neighbors
        embedding_chunk = self.embedding_[
            chunk_start : chunk_start + chunk_size
        ]  # (chunk_size, n_components)
        embedding_neighbors = self.embedding_[
            indices_chunk
        ]  # (chunk_size, n_neighbors, n_components)

        # Compute Student-t kernel directly: 1 / (1 + ||x - y||^2)
        distances_sq = torch.sum(
            (embedding_chunk.unsqueeze(1) - embedding_neighbors) ** 2, dim=-1
        )
        Q = 1.0 / (1.0 + distances_sq)  # Student-t with df=1

        # Apply the same transformation as in LargeVis
        Q = Q / (Q + 1)

        # Compute cross-entropy loss for this chunk
        # The affinity_in_ already contains only this GPU's chunk values
        return cross_entropy_loss(self.affinity_in_, Q)

    def _compute_repulsive_loss(self):
        """Compute repulsive loss with negative sampling.

        Each GPU computes repulsion for its chunk's points against global negatives.
        The total loss is automatically summed across GPUs via autograd.
        """
        # Get chunk boundaries for this GPU
        if hasattr(self.affinity_in, "chunk_start_"):
            chunk_start = self.affinity_in.chunk_start_
            chunk_size = self.affinity_in.chunk_size_
        elif hasattr(self, "chunk_start_"):
            chunk_start = self.chunk_start_
            chunk_size = self.chunk_size_
        else:
            # Fallback: assume full dataset if no chunking
            chunk_start = 0
            chunk_size = self.n_samples_in_

        # Extract embeddings for this chunk and its negative samples
        embedding_chunk = self.embedding_[
            chunk_start : chunk_start + chunk_size
        ]  # (chunk_size, n_components)
        # neg_indices_ has shape (chunk_size, n_negatives) - contains global indices
        embedding_negatives = self.embedding_[
            self.neg_indices_
        ]  # (chunk_size, n_negatives, n_components)

        # Compute Student-t kernel directly: 1 / (1 + ||x - y||^2)
        distances_sq = torch.sum(
            (embedding_chunk.unsqueeze(1) - embedding_negatives) ** 2, dim=-1
        )
        Q = 1.0 / (1.0 + distances_sq)  # Student-t with df=1

        # Apply the same transformation as in LargeVis
        Q = Q / (Q + 1)

        # Compute repulsive loss for this chunk
        # Division by n_samples_in_ normalizes by total dataset size
        return -sum_red((1 - Q).log(), dim=(0, 1)) / self.n_samples_in_

    def _training_step(self):
        """Override training step to handle multi-GPU gradient synchronization.

        After computing gradients, use all_reduce to sum them across GPUs.
        """
        self.optimizer_.zero_grad(set_to_none=True)

        # Compute loss for this GPU's chunk
        loss = self._compute_loss()
        loss.backward()

        # Synchronize gradients across GPUs if multi-GPU
        if self.is_multi_gpu:
            # All-reduce (sum) gradients across all GPUs
            # This mimics single GPU where all gradients are naturally summed
            dist.all_reduce(self.embedding_.grad)

        # Apply gradient update
        self.optimizer_.step()
        if self.scheduler_ is not None:
            self.scheduler_.step()

        return loss

    def _init_embedding(self, X: torch.Tensor):
        """Initialize embedding for multi-GPU setting.

        Rank 0 initializes, then broadcasts to all other ranks.

        Parameters
        ----------
        X : torch.Tensor
            Input data (used for PCA initialization if needed).
        """
        if self.is_multi_gpu:
            rank = dist.get_rank()
            current_device = torch.cuda.current_device()

            if rank == 0:
                # Only rank 0 initializes the embedding
                super()._init_embedding(X)
            else:
                # Other ranks allocate empty tensor
                n = X.shape[0]
                self.embedding_ = torch.empty(
                    (n, self.n_components),
                    device=f"cuda:{current_device}",
                    dtype=X.dtype,
                    requires_grad=True,
                )

            # Ensure tensor is contiguous before broadcast
            if not self.embedding_.is_contiguous():
                self.embedding_ = self.embedding_.contiguous()

            # Broadcast from rank 0 to all ranks (in-place)
            dist.broadcast(self.embedding_, src=0)

            # After broadcast, we need to recreate as a leaf tensor for optimization
            # Detach and recreate to make it a leaf variable
            self.embedding_ = self.embedding_.detach().requires_grad_(True)
        else:
            # Standard single-GPU initialization
            super()._init_embedding(X)

    def on_affinity_computation_end(self):
        """Prepare for negative sampling in multi-GPU setting."""
        if not self.is_multi_gpu:
            # Use parent class implementation for single GPU
            return super().on_affinity_computation_end()

        # Get this GPU's chunk information
        chunk_start = self.affinity_in.chunk_start_
        chunk_size = self.affinity_in.chunk_size_
        current_device = torch.cuda.current_device()

        # Create self indices for this chunk
        self_idxs = torch.arange(
            chunk_start, chunk_start + chunk_size, device=f"cuda:{current_device}"
        ).unsqueeze(1)

        if self.discard_NNs and hasattr(self.affinity_in, "indices_"):
            # Exclude self and nearest neighbors from negative sampling
            # Note: indices_ are global indices, not chunk-relative
            exclude = torch.cat([self_idxs, self.affinity_in.indices_], dim=1)
        else:
            exclude = self_idxs

        exclude_sorted, _ = exclude.sort(dim=1)
        self.register_buffer(
            "negative_exclusion_indices_", exclude_sorted, persistent=False
        )

        # Store chunk info for negative sampling
        self.chunk_start_ = chunk_start
        self.chunk_size_ = chunk_size

    def on_training_step_start(self):
        """Sample negatives for this GPU's chunk at each training step."""
        if not self.is_multi_gpu:
            # Use parent class implementation for single GPU
            return super().on_training_step_start()

        # Skip parent's negative sampling, do our own
        super(SampledNeighborEmbedding, self).on_training_step_start()

        current_device = torch.cuda.current_device()

        # Fast path for k=1 (only excluding self-indices)
        if self.negative_exclusion_indices_.shape[1] == 1:
            # Sample from [0, n-1) for each point in the chunk
            negatives = torch.randint(
                0,
                self.n_samples_in_ - 1,
                (self.chunk_size_, self.n_negatives),
                device=f"cuda:{current_device}",
            )
            # Global indices for points in this chunk
            global_self_idx = torch.arange(
                self.chunk_start_,
                self.chunk_start_ + self.chunk_size_,
                device=f"cuda:{current_device}",
            ).unsqueeze(1)
            # Shift indices >= global_self_idx by 1 to skip self
            neg_indices = negatives + (negatives >= global_self_idx).long()
        else:
            # General case: use searchsorted for multiple exclusions
            # Sample negatives for all points in chunk
            negatives = torch.randint(
                1,
                self.n_samples_in_ - self.negative_exclusion_indices_.shape[1],
                (self.chunk_size_, self.n_negatives),
                device=f"cuda:{current_device}",
            )

            # Apply searchsorted row-wise
            # negative_exclusion_indices_ has shape (chunk_size, num_exclusions)
            # negatives has shape (chunk_size, n_negatives)
            # We need to apply searchsorted for each row independently
            neg_indices = torch.zeros_like(negatives)
            for i in range(self.chunk_size_):
                shifts = torch.searchsorted(
                    self.negative_exclusion_indices_[i], negatives[i], right=True
                )
                neg_indices[i] = negatives[i] + shifts

        self.register_buffer("neg_indices_", neg_indices, persistent=False)
