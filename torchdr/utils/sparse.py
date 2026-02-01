import torch
import torch.distributed as dist
from typing import Tuple, Literal
from torchdr.distributed import DistributedContext


def flatten_sparse(
    values: torch.Tensor, indices: torch.LongTensor
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """Flatten sparse representation into i, j, v.

    Parameters
    ----------
    values : torch.Tensor
        Dense tensor of shape (n, k) containing non-zero values.
    indices : torch.LongTensor
        Long tensor of shape (n, k) containing column indices.

    Returns
    -------
    i : torch.LongTensor
        Flat row indices of length N.
    j : torch.LongTensor
        Flat column indices of length N.
    v : torch.Tensor
        Flat values of length N.
    """
    n, k = values.shape
    device = values.device

    rows = torch.arange(n, device=device).unsqueeze(1).expand(n, k)
    i = rows.reshape(-1)
    j = indices.reshape(-1)
    v = values.reshape(-1)
    return i, j, v


def merge_symmetry(
    i: torch.LongTensor, j: torch.LongTensor, v: torch.Tensor, n: int
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
    """Merge P and Pᵀ entries by unique coordinates.

    Parameters
    ----------
    i : torch.LongTensor
        Row indices of P entries.
    j : torch.LongTensor
        Column indices of P entries.
    v : torch.Tensor
        Values of P entries.
    n : int
        Number of rows/columns of the square matrix P.

    Returns
    -------
    i_out : torch.LongTensor
        Unique row indices of combined entries.
    j_out : torch.LongTensor
        Unique column indices of combined entries.
    vP : torch.Tensor
        Sum of P entries at each unique position.
    vPT : torch.Tensor
        Sum of Pᵀ entries at each unique position.
    """
    keys_P = i * n + j
    keys_PT = j * n + i

    # Combine and find unique positions
    keys_all = torch.cat([keys_P, keys_PT], dim=0)
    vals_all = torch.cat([v, v], dim=0)
    mask_P = torch.arange(keys_all.numel(), device=keys_all.device) < v.numel()

    uniq_keys, inv_idx = torch.unique(keys_all, sorted=True, return_inverse=True)
    M = uniq_keys.numel()

    # Scatter-add contributions from P vs. Pᵀ
    dtype, device = v.dtype, v.device
    vP = torch.zeros(M, dtype=dtype, device=device)
    vPT = torch.zeros(M, dtype=dtype, device=device)
    vP.scatter_add_(0, inv_idx, vals_all * mask_P.to(dtype))
    vPT.scatter_add_(0, inv_idx, vals_all * (~mask_P).to(dtype))

    # Decode back to (i,j)
    i_out = uniq_keys // n
    j_out = uniq_keys % n
    return i_out, j_out, vP, vPT


def pack_to_rowwise(
    i_out: torch.LongTensor, j_out: torch.LongTensor, v_out: torch.Tensor, n: int
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Pack flat entries back into padded row-wise format.

    Parameters
    ----------
    i_out : torch.LongTensor
        Row indices of combined entries.
    j_out : torch.LongTensor
        Column indices of combined entries.
    v_out : torch.Tensor
        Values of combined entries.
    n : int
        Number of rows of the matrix.

    Returns
    -------
    values_out : torch.Tensor
        Padded values tensor of shape (n, k_out).
    indices_out : torch.LongTensor
        Padded indices tensor of shape (n, k_out), with -1 for unused slots.
    """
    # Handle empty case
    if i_out.numel() == 0:
        return torch.zeros((n, 0), dtype=v_out.dtype, device=v_out.device), torch.zeros(
            (n, 0), dtype=torch.long, device=v_out.device
        )

    counts = torch.bincount(i_out, minlength=n)
    max_k_out = counts.max().item()

    if max_k_out == 0:
        return torch.zeros((n, 0), dtype=v_out.dtype, device=v_out.device), torch.zeros(
            (n, 0), dtype=torch.long, device=v_out.device
        )

    values_out = torch.zeros((n, max_k_out), dtype=v_out.dtype, device=v_out.device)
    indices_out = torch.full((n, max_k_out), -1, dtype=torch.long, device=v_out.device)

    # More efficient slot computation using cumsum
    row_offsets = torch.zeros(n + 1, dtype=torch.long, device=v_out.device)
    row_offsets[1:] = counts.cumsum(0)

    # Compute slots within each row directly
    slots = torch.arange(i_out.numel(), device=v_out.device) - row_offsets[i_out]
    flat_pos = i_out * max_k_out + slots

    values_out.view(-1).scatter_(0, flat_pos, v_out)
    indices_out.view(-1).scatter_(0, flat_pos, j_out)

    return values_out, indices_out


def _combine_P_PT(
    vP: torch.Tensor, vPT: torch.Tensor, mode: Literal["sum", "sum_minus_prod"]
) -> torch.Tensor:
    """Combine P and P^T values based on mode.

    Parameters
    ----------
    vP : torch.Tensor
        Values from P matrix.
    vPT : torch.Tensor
        Values from P^T matrix.
    mode : {"sum", "sum_minus_prod"}
        Combination mode.

    Returns
    -------
    v_combined : torch.Tensor
        Combined values.
    """
    if mode == "sum":
        return vP + vPT
    elif mode == "sum_minus_prod":
        return vP + vPT - vP * vPT
    else:
        raise ValueError(f"Unsupported mode {mode!r}")


def symmetrize_sparse(
    values: torch.Tensor,
    indices: torch.LongTensor,
    mode: Literal["sum", "sum_minus_prod"] = "sum_minus_prod",
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Symmetrize sparse matrix P per mode.

    Parameters
    ----------
    values : torch.Tensor
        Dense tensor of shape (n, k) for P's non-zero values.
    indices : torch.LongTensor
        Long tensor of shape (n, k) for P's column indices.
    mode : {"sum", "sum_minus_prod"}, optional
        - "sum": compute Q = P + Pᵀ
        - "sum_minus_prod": compute Q = P + Pᵀ - P∘Pᵀ (default)

    Returns
    -------
    values_out : torch.Tensor
        Padded values of Q with shape (n, k_out).
    indices_out : torch.LongTensor
        Padded column indices of Q with shape (n, k_out).
    """
    n, _ = values.shape

    # 1) flatten sparse P
    i, j, v = flatten_sparse(values, indices)

    # 2) merge P and Pᵀ entries
    i_out, j_out, vP, vPT = merge_symmetry(i, j, v, n)

    # 3) compute final values using shared helper
    v_out = _combine_P_PT(vP, vPT, mode)

    # 4) pack back to padded row-wise format
    return pack_to_rowwise(i_out, j_out, v_out, n)


def symmetrize_sparse_cpu_offload(
    values: torch.Tensor,
    indices: torch.LongTensor,
    mode: Literal["sum", "sum_minus_prod"] = "sum_minus_prod",
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Symmetrize sparse matrix with CPU offload for large-scale data.

    Moves data to CPU for memory-intensive operations, then back to original device.
    Use this when GPU memory is insufficient for the standard symmetrization.

    Parameters
    ----------
    values : torch.Tensor
        Dense tensor of shape (n, k) for P's non-zero values.
    indices : torch.LongTensor
        Long tensor of shape (n, k) for P's column indices.
    mode : {"sum", "sum_minus_prod"}, optional
        - "sum": compute Q = P + Pᵀ
        - "sum_minus_prod": compute Q = P + Pᵀ - P∘Pᵀ (default)

    Returns
    -------
    values_out : torch.Tensor
        Padded values of Q with shape (n, k_out), on the original device.
    indices_out : torch.LongTensor
        Padded column indices of Q with shape (n, k_out), on the original device.
    """
    original_device = values.device

    # Move to CPU for memory-intensive operations
    values_cpu = values.cpu()
    indices_cpu = indices.cpu()

    # Perform symmetrization on CPU
    values_out, indices_out = symmetrize_sparse(values_cpu, indices_cpu, mode)

    # Move results back to original device
    return values_out.to(original_device), indices_out.to(original_device)


def distributed_symmetrize_sparse(
    values: torch.Tensor,
    indices: torch.LongTensor,
    chunk_start: int,
    chunk_size: int,
    n_total: int,
    mode: Literal["sum", "sum_minus_prod"] = "sum_minus_prod",
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Symmetrize sparse affinity matrix in distributed multi-GPU setting.

    Each GPU owns a chunk of rows and exchanges edges with other GPUs
    to properly symmetrize the affinity matrix based on the specified mode.

    Parameters
    ----------
    values : torch.Tensor
        Affinity values of shape (chunk_size, k).
    indices : torch.LongTensor
        Column indices of shape (chunk_size, k).
    chunk_start : int
        Starting row index for this GPU's chunk.
    chunk_size : int
        Number of rows in this chunk.
    n_total : int
        Total number of rows/columns in the full matrix.
    mode : {"sum", "sum_minus_prod"}
        How to combine P and P^T:
        - "sum": compute Q = P + P^T
        - "sum_minus_prod": compute Q = P + P^T - P∘P^T (default)

    Returns
    -------
    values_sym : torch.Tensor
        Symmetrized affinity values.
    indices_sym : torch.LongTensor
        Column indices of symmetrized affinities.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "distributed_symmetrize requires torch.distributed to be initialized"
        )

    world_size = dist.get_world_size()
    device = values.device

    # Step 1: Flatten local edges to (i, j, v) format
    i, j, v = flatten_sparse(values, indices)
    i = i + chunk_start  # Convert to global indices

    # Step 2: Sort edges by target rank for efficient packing
    target_ranks = DistributedContext.get_rank_for_indices(j, n_total, world_size)

    # Sort by target rank for contiguous memory access
    sorted_idx = torch.argsort(target_ranks)
    i_sorted = i[sorted_idx]
    j_sorted = j[sorted_idx]
    v_sorted = v[sorted_idx]
    target_sorted = target_ranks[sorted_idx]

    # Step 3: Vectorized packing of edges for each rank
    send_tensors = [
        torch.empty((3, 0), device=device, dtype=torch.float32)
        for _ in range(world_size)
    ]

    if target_sorted.numel() > 0:
        # Find boundaries for each rank's edges
        unique_targets, counts = torch.unique_consecutive(
            target_sorted, return_counts=True
        )
        offsets = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)])

        # Pack edges for each rank that has data
        for idx, r in enumerate(unique_targets.tolist()):
            start = offsets[idx].item()
            end = offsets[idx + 1].item()
            if end > start and r < world_size:
                send_tensors[r] = torch.stack(
                    [
                        i_sorted[start:end].float(),
                        j_sorted[start:end].float(),
                        v_sorted[start:end],
                    ],
                    dim=0,
                ).contiguous()

    # Step 4: Exchange sizes first to allocate correct receive buffers
    send_sizes = torch.tensor(
        [s.shape[1] for s in send_tensors], device=device, dtype=torch.long
    )
    recv_sizes = torch.zeros(world_size, device=device, dtype=torch.long)
    dist.all_to_all_single(recv_sizes, send_sizes)

    # Step 5: Allocate receive buffers with correct sizes
    recv_tensors = []
    for r in range(world_size):
        size = recv_sizes[r].item()
        recv_tensors.append(torch.zeros((3, size), device=device, dtype=torch.float32))

    # Step 6: All-to-all exchange with properly sized buffers
    dist.all_to_all(recv_tensors, send_tensors)

    # Step 7: Unpack received edges (these are edges where we own the transpose)
    if any(t.shape[1] > 0 for t in recv_tensors):
        recv_all = torch.cat([t for t in recv_tensors if t.shape[1] > 0], dim=1)
        recv_i = recv_all[0].long()
        recv_j = recv_all[1].long()
        recv_v = recv_all[2]

        # Combine with local edges for symmetrization
        # Local edges: (i, j) with values v
        # Received edges need to be transposed: (j, i) becomes part of our P^T
        all_i = torch.cat([i, recv_j])
        all_j = torch.cat([j, recv_i])
        all_v = torch.cat([v, recv_v])
    else:
        all_i = i
        all_j = j
        all_v = v

    # Step 8: Move to CPU for memory-intensive merge_symmetry operation
    # This reduces GPU memory by ~70% for large-scale data
    all_i_cpu = all_i.cpu()
    all_j_cpu = all_j.cpu()
    all_v_cpu = all_v.cpu()

    # Apply merge_symmetry on CPU to handle duplicates
    i_sym, j_sym, vP, vPT = merge_symmetry(all_i_cpu, all_j_cpu, all_v_cpu, n_total)

    # Step 9: Combine P and P^T using shared helper (still on CPU)
    v_sym = _combine_P_PT(vP, vPT, mode)

    # Step 10: Filter to keep only edges in our chunk (still on CPU)
    mask = (i_sym >= chunk_start) & (i_sym < chunk_start + chunk_size)
    i_local = i_sym[mask] - chunk_start
    j_local = j_sym[mask]
    v_local = v_sym[mask]

    # Step 11: Pack to row-wise format and move back to GPU
    values_out, indices_out = pack_to_rowwise(i_local, j_local, v_local, chunk_size)
    return values_out.to(device), indices_out.to(device)
