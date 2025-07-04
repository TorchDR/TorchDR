import torch
from typing import Tuple, Literal


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
    counts = torch.bincount(i_out, minlength=n)
    max_k_out = counts.max()

    values_out = torch.zeros((n, max_k_out), dtype=v_out.dtype, device=v_out.device)
    indices_out = torch.full((n, max_k_out), -1, dtype=torch.long, device=v_out.device)

    M_all = i_out.numel()
    pos = torch.arange(M_all, device=v_out.device)

    is_new = torch.cat(
        [torch.tensor([True], device=v_out.device), i_out[1:] != i_out[:-1]]
    )
    row_starts = torch.nonzero(is_new, as_tuple=False).flatten()

    grp = torch.searchsorted(row_starts, pos, right=True) - 1
    slot = pos - row_starts[grp]
    flat_pos = i_out * max_k_out + slot

    values_out.view(-1).scatter_(0, flat_pos, v_out)
    indices_out.view(-1).scatter_(0, flat_pos, j_out)

    return values_out, indices_out


def sym_sparse_op(
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

    # 3) compute final values inline
    if mode == "sum":
        v_out = vP + vPT
    elif mode == "sum_minus_prod":
        v_out = vP + vPT - vP * vPT
    else:
        raise ValueError(f"Unsupported mode {mode!r}")

    # 4) pack back to padded row-wise format
    return pack_to_rowwise(i_out, j_out, v_out, n)
