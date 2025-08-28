"""Multi-GPU implementation of LargeVis."""

import os
import torch
import torch.distributed as dist
from typing import Optional, Union

from torchdr.neighbor_embedding.base import SampledNeighborEmbedding
from torchdr.affinity.multi_gpu import EntropicAffinityMultiGPU
from torchdr.utils import cross_entropy_loss, sum_red


class LargeVisMultiGPU(SampledNeighborEmbedding):
    """Multi-GPU implementation of LargeVis for distributed training with torchrun.

    Each GPU processes its chunk of the affinity matrix independently.
    Embeddings are replicated across all GPUs for loss computation.
    Must be launched with torchrun for distributed execution.

    Examples
    --------
    Using with torchrun::

        # In script.py:
        model = LargeVisMultiGPU(perplexity=30)
        model.fit(X)

        # Launch with:
        # torchrun --nproc_per_node=4 script.py
    """

    def __init__(
        self,
        perplexity: float = 30,
        n_components: int = 2,
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
        compile: bool = False,
        gradient_compression: str = None,
    ):
        """Initialize multi-GPU LargeVis for distributed training.

        Must be launched with torchrun for distributed execution.

        Parameters
        ----------
        perplexity : float, default=30
            Number of effective nearest neighbors.
        n_components : int, default=2
            Dimension of the embedding space.
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
        compile : bool, default=False
            Whether to use torch.compile.
        gradient_compression : str, optional
            Gradient compression for faster all_reduce communication.
            Options: None (no compression), "fp16", "bf16".
            bf16 maintains better numerical stability than fp16.
        """
        self.gradient_compression = gradient_compression
        # Check that we're in distributed mode (launched with torchrun)
        if not dist.is_initialized():
            raise RuntimeError(
                "[TorchDR] LargeVisMultiGPU requires distributed mode. "
                "Launch with torchrun, e.g.: torchrun --nproc_per_node=<num_gpus> script.py"
            )

        # In distributed mode, each process handles one GPU
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # Use local rank for device assignment
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        self.is_multi_gpu = self.world_size > 1

        # Store LargeVis-specific parameters
        self.perplexity = perplexity
        self.n_negatives = n_negatives
        self.sparsity = sparsity
        # Always False for multi-GPU to simplify negative sampling
        self.discard_NNs = False

        affinity_in = EntropicAffinityMultiGPU(
            perplexity=perplexity,
            verbose=verbose,
        )

        # Call parent's __init__ (SampledNeighborEmbedding)
        super().__init__(
            affinity_in=affinity_in,
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
            discard_NNs=False,  # Always False for multi-GPU
            check_interval=check_interval,
            compile=compile,
        )

    def _get_chunk_info(self):
        """Get chunk start and size for current rank.

        Returns
        -------
        chunk_start : int
            Starting index for this rank's chunk
        chunk_size : int
            Size of this rank's chunk
        """
        if hasattr(self.affinity_in, "chunk_start_"):
            return self.affinity_in.chunk_start_, self.affinity_in.chunk_size_
        elif hasattr(self, "chunk_start_"):
            return self.chunk_start_, self.chunk_size_
        else:
            return 0, self.n_samples_in_

    def _compute_attractive_loss(self):
        """Compute attractive loss using chunked affinity matrix.

        Each GPU computes the loss for its chunk of the affinity matrix.
        The embedding is replicated across all GPUs.
        """
        chunk_start, chunk_size = self._get_chunk_info()

        # Get this rank's chunk of the embedding
        embedding_chunk = self.embedding_[chunk_start : chunk_start + chunk_size]

        # NN_indices_ contains global indices of neighbors for this chunk
        # Shape: NN_indices_ is [chunk_size, k], embedding_neighbors will be [chunk_size, k, n_components]
        embedding_neighbors = self.embedding_[self.NN_indices_]

        # Compute pairwise distances between chunk points and their neighbors
        # embedding_chunk: [chunk_size, n_components] -> [chunk_size, 1, n_components]
        # embedding_neighbors: [chunk_size, k, n_components]
        distances_sq = torch.sum(
            (embedding_chunk.unsqueeze(1) - embedding_neighbors) ** 2, dim=-1
        )
        Q = 1.0 / (1.0 + distances_sq)
        Q = Q / (Q + 1)
        return cross_entropy_loss(self.affinity_in_, Q)

    def _compute_repulsive_loss(self):
        """Compute repulsive loss with negative sampling.

        Each GPU computes repulsion for its chunk's points against global negatives.
        The total loss is automatically summed across GPUs via autograd.
        """
        chunk_start, chunk_size = self._get_chunk_info()

        embedding_chunk = self.embedding_[chunk_start : chunk_start + chunk_size]

        # Note: We can't use pairwise_distances with indices parameter here because
        # neg_indices_ contains global indices, not indices relative to embedding_chunk
        embedding_negatives = self.embedding_[self.neg_indices_]
        distances_sq = torch.sum(
            (embedding_chunk.unsqueeze(1) - embedding_negatives) ** 2, dim=-1
        )
        Q = 1.0 / (1.0 + distances_sq)
        Q = Q / (Q + 1)
        return -sum_red((1 - Q).log(), dim=(0, 1)) / self.n_samples_in_

    def _training_step(self):
        """Override training step to handle multi-GPU gradient synchronization.

        After computing gradients, use all_reduce to sum them across GPUs.
        Optionally uses gradient compression for faster communication.
        """
        self.optimizer_.zero_grad(set_to_none=True)

        loss = self._compute_loss()
        loss.backward()

        # Synchronize gradients across all ranks if multi-GPU
        if self.world_size > 1:
            if self.gradient_compression == "fp16":
                # Create fp16 copy for communication
                grad_compressed = self.embedding_.grad.half()
                dist.all_reduce(grad_compressed, op=dist.ReduceOp.SUM)
                # Copy back to original gradient tensor (preserves dtype)
                self.embedding_.grad.copy_(grad_compressed)
            elif self.gradient_compression == "bf16":
                # Create bf16 copy for communication
                grad_compressed = self.embedding_.grad.bfloat16()
                dist.all_reduce(grad_compressed, op=dist.ReduceOp.SUM)
                # Copy back to original gradient tensor (preserves dtype)
                self.embedding_.grad.copy_(grad_compressed)
            else:
                # No compression
                dist.all_reduce(self.embedding_.grad, op=dist.ReduceOp.SUM)

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
        if self.world_size > 1:
            if self.rank == 0:
                super()._init_embedding(X)
            else:
                n = X.shape[0]
                self.embedding_ = torch.empty(
                    (n, self.n_components),
                    device=self.device,
                    dtype=X.dtype,
                    requires_grad=True,
                )

            if not self.embedding_.is_contiguous():
                self.embedding_ = self.embedding_.contiguous()

            dist.broadcast(self.embedding_, src=0)

            self.embedding_ = self.embedding_.detach().requires_grad_(True)
        else:
            super()._init_embedding(X)

    def on_affinity_computation_end(self):
        """Prepare for negative sampling in multi-GPU setting."""
        if self.world_size == 1:
            return super().on_affinity_computation_end()

        # Store chunk info for negative sampling
        self.chunk_start_ = self.affinity_in.chunk_start_
        self.chunk_size_ = self.affinity_in.chunk_size_

    def on_training_step_start(self):
        """Sample negatives for this GPU's chunk at each training step."""
        if self.world_size == 1:
            return super().on_training_step_start()

        super(SampledNeighborEmbedding, self).on_training_step_start()

        # Simple negative sampling without excluding NNs
        negatives = torch.randint(
            0,
            self.n_samples_in_ - 1,
            (self.chunk_size_, self.n_negatives),
            device=self.device,
        )

        # Adjust for self-exclusion only
        global_self_idx = torch.arange(
            self.chunk_start_,
            self.chunk_start_ + self.chunk_size_,
            device=self.device,
        ).unsqueeze(1)

        neg_indices = negatives + (negatives >= global_self_idx).long()

        self.register_buffer("neg_indices_", neg_indices, persistent=False)
