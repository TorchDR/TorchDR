"""Distances based on Faiss backend."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
import warnings
from typing import Union

from torchdr.utils.faiss import faiss

LIST_METRICS_FAISS = ["euclidean", "sqeuclidean", "angular"]


def pairwise_distances_faiss(
    X: torch.Tensor,
    k: Union[int, torch.Tensor],
    Y: torch.Tensor = None,
    metric: str = "sqeuclidean",
    exclude_diag: bool = False,
):
    r"""Compute the k nearest neighbors using FAISS.

    Supported metrics are:
      - "euclidean": returns the Euclidean distance (square root of the squared distance)
      - "sqeuclidean": returns the squared Euclidean distance (as computed by FAISS)
      - "angular": returns the negative inner-product (after normalizing vectors)

    If Y is not provided then we assume a self–search and, if `exclude_diag` is True,
    the self–neighbor is removed from the results.

    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
        Query dataset.
    Y : torch.Tensor of shape (m, d), optional
        Database dataset. If None, Y is set equal to X.
    metric : str, default "euclidean"
        One of "euclidean", "sqeuclidean", or "angular".
    k : int or torch.Tensor, optional
        Number of nearest neighbors to return. If tensor, will be converted to int.
        (If `exclude_diag` is True in a self–search, then k+1 neighbors are retrieved first.)
    exclude_diag : bool, default False
        When True and Y is not provided (i.e. self–search), the self–neighbor (index i for query i)
        is excluded from the k results.

    Returns
    -------
    distances : torch.Tensor of shape (n, k)
        Nearest neighbor distances.
        For metric=="euclidean", distances are Euclidean (i.e. square root of L2^2).
        For metric=="sqeuclidean", distances are the squared Euclidean distances.
        For metric=="angular", distances are the (normalized) inner product scores.
    indices : torch.Tensor of shape (n, k)
        Indices of the k nearest neighbors.
    """
    if metric not in LIST_METRICS_FAISS:
        raise ValueError(
            "[TorchDR] Only 'euclidean', 'sqeuclidean', and 'angular' metrics "
            "are supported for FAISS."
        )

    # Convert k to integer if it's a tensor (for FAISS compatibility)
    if isinstance(k, torch.Tensor):
        k = int(k.item())
    else:
        k = int(k)

    # Convert input tensor X to a NumPy array.
    dtype = X.dtype
    X_np = X.detach().cpu().numpy().astype(np.float32)
    n, d = X_np.shape

    # If Y is not provided, reuse X_np for Y_np.
    if Y is None or Y is X:
        Y_np = X_np
        do_exclude = exclude_diag
    else:
        Y_np = Y.detach().cpu().numpy().astype(np.float32)
        do_exclude = False

    # Set up the FAISS index depending on the metric.
    if metric == "angular":
        # Use the inner product index. Note: FAISS returns negative inner products.
        index = faiss.IndexFlatIP(d)

    elif metric in {"euclidean", "sqeuclidean"}:
        # Use the L2 index. Note: FAISS returns squared distances.
        index = faiss.IndexFlatL2(d)

    else:
        # This branch should never be reached due to the initial check.
        raise ValueError(f"[TorchDR] ERROR : Metric '{metric}' is not supported.")

    device = X.device
    if device.type == "cuda":
        if hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            warnings.warn(
                "[TorchDR] WARNING: `faiss-gpu` not installed, using CPU for Faiss computations. "
                "This may be slow. For faster performance, install `faiss-gpu`."
            )

    # Add the database vectors to the index.
    index.add(Y_np)

    # If self-search and excluding self, search for one extra neighbor.
    if do_exclude:
        k_search = k + 1
    else:
        k_search = k

    # Perform the search.
    D, Ind = index.search(X_np, k_search)  # D: (n, k_search), I: (n, k_search)

    # For "euclidean", take the square root of the squared distances.
    if metric == "euclidean":
        D = np.sqrt(D)
    # For "angular", negate the inner products.
    elif metric == "angular":
        D = -D
    # For "sqeuclidean", leave the distances as returned (i.e. squared).

    # If doing self–search with self–exclusion, remove the self neighbor from the results.
    if do_exclude:
        # In a self-search, FAISS returns the query point itself as the first neighbor.
        # We searched for k+1 neighbors, so we discard the first column.
        D = D[:, 1:]
        Ind = Ind[:, 1:]

    # Convert back to torch tensors.
    distances = torch.from_numpy(D).to(device).to(dtype)
    indices = torch.from_numpy(Ind).to(device).long()

    return distances, indices
