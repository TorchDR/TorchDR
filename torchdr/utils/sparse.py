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


def _merge_sparse_keys(
    keys_P: torch.LongTensor,
    values_P: torch.Tensor,
    keys_PT: torch.LongTensor,
    values_PT: torch.Tensor,
    n: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
    """Merge separate P and Pᵀ entries by encoded matrix coordinates."""
    n_P = keys_P.numel()
    keys_all = torch.cat((keys_P, keys_PT))
    del keys_P, keys_PT

    unique_keys, inverse = torch.unique(keys_all, sorted=True, return_inverse=True)
    del keys_all

    n_unique = unique_keys.numel()
    values_P_out = torch.zeros(n_unique, dtype=values_P.dtype, device=values_P.device)
    values_PT_out = torch.zeros_like(values_P_out)
    values_P_out.scatter_add_(0, inverse[:n_P], values_P)
    values_PT_out.scatter_add_(0, inverse[n_P:], values_PT)
    del inverse

    i_out = torch.div(unique_keys, n, rounding_mode="floor")
    j_out = unique_keys.remainder(n)
    return i_out, j_out, values_P_out, values_PT_out


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
    return _merge_sparse_keys(i * n + j, v, j * n + i, v, n)


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

    Notes
    -----
    Edge exchange is performed on the input device with ``all_to_all_single``
    over flat buffers, which both NCCL and the Gloo CPU backend support.
    Coalescing and row-wise packing are then performed on CPU to reduce peak
    accelerator memory. The returned tensors are moved back to the input device.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "distributed_symmetrize requires torch.distributed to be initialized"
        )

    world_size = dist.get_world_size()
    device = values.device

    # Step 1: Flatten and encode local edges.
    i, j, v = flatten_sparse(values, indices)
    i.add_(chunk_start)
    keys = i * n_total + j

    # Step 2: Sort encoded edges by the rank that owns their target row.
    target_ranks = DistributedContext.get_rank_for_indices(j, n_total, world_size)
    sorted_idx = torch.argsort(target_ranks)
    keys_sorted = keys[sorted_idx]
    values_sorted = v[sorted_idx]
    target_sorted = target_ranks[sorted_idx]
    del i, j, target_ranks

    # Step 3: Derive the per-rank split sizes. The edges are already ordered by
    # owning rank, so the boundaries are just the insertion points of the rank
    # labels and the sorted buffers double as flat send buffers.
    boundaries = torch.arange(world_size + 1, device=device)
    send_offsets = torch.searchsorted(target_sorted, boundaries)
    send_counts = send_offsets[1:] - send_offsets[:-1]
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()
    del boundaries, send_offsets, send_counts, recv_counts, target_sorted, sorted_idx

    # Step 4: Exchange exact integer coordinates and values without dtype casts.
    # Keeping encoded coordinates as int64 avoids the precision loss caused by
    # packing indices as float32. One contiguous buffer per payload keeps the
    # allocation and host-synchronization count independent of the world size
    # and, unlike the list-based all_to_all, is supported by the Gloo backend.
    n_received = sum(recv_splits)
    transpose_keys = torch.empty(n_received, device=device, dtype=torch.long)
    transpose_values = torch.empty(n_received, device=device, dtype=v.dtype)
    dist.all_to_all_single(transpose_keys, keys_sorted, recv_splits, send_splits)
    dist.all_to_all_single(transpose_values, values_sorted, recv_splits, send_splits)
    del keys_sorted, values_sorted, send_splits, recv_splits

    # Step 5: Offload local and received edges before the memory-heavy unique.
    local_keys = keys.cpu()
    local_values = v.cpu()
    del keys, v
    transpose_keys = transpose_keys.cpu()
    transpose_values = transpose_values.cpu()

    # Received keys encode original (i, j) entries. Rewrite them in place as
    # (j, i), retaining their provenance as Pᵀ rather than treating them as P.
    received_rows = torch.div(transpose_keys, n_total, rounding_mode="floor")
    transpose_keys.remainder_(n_total).mul_(n_total).add_(received_rows)
    del received_rows

    # Step 6: Coalesce P and Pᵀ separately. This is both correct for reciprocal
    # edges and avoids duplicating the already combined edge list a second time.
    i_sym, j_sym, vP, vPT = _merge_sparse_keys(
        local_keys,
        local_values,
        transpose_keys,
        transpose_values,
        n_total,
    )
    del local_keys, local_values, transpose_keys, transpose_values

    # Step 7: Combine components on CPU and pack the rows owned by this rank.
    v_sym = _combine_P_PT(vP, vPT, mode)
    del vP, vPT
    local_mask = (i_sym >= chunk_start) & (i_sym < chunk_start + chunk_size)
    i_local = i_sym[local_mask].sub_(chunk_start)
    j_local = j_sym[local_mask]
    v_local = v_sym[local_mask]
    values_out, indices_out = pack_to_rowwise(i_local, j_local, v_local, chunk_size)

    # Step 8: Restore the caller's device contract.
    return values_out.to(device), indices_out.to(device)
