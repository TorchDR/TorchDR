"""Affinity matrix used in PACMAP."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Union, Optional

import numpy as np
import torch

from torchdr.affinity.base import SparseLogAffinity
from torchdr.utils import kmin, check_neighbor_param


class PACMAPAffinity(SparseLogAffinity):
    r"""Compute the input affinity used in PACMAP :cite:`wang2021understanding`.

    Parameters
    ----------
    n_neighbors : float, optional
        Number of effective nearest neighbors to consider. Similar to the perplexity.
    tol : float, optional
        Precision threshold for the root search.
    metric : str, optional
        Metric to use for pairwise distances computation.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, optional
        Verbosity. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 10,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
    ):
        self.n_neighbors = n_neighbors

        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=True,  # PACMAP uses sparsity mode
        )

    def _compute_sparse_log_affinity(self, X: Union[torch.Tensor, np.ndarray]):
        r"""Compute the input affinity matrix of PACMAP from input data X.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : PACMAPAffinityIn
            The fitted instance.
        """
        if self.verbose:
            print("[TorchDR] Affinity : computing the input affinity matrix of PACMAP.")

        n_samples_in = X.shape[0]
        k = min(self.n_neighbors + 50, n_samples_in)
        k = check_neighbor_param(k, n_samples_in)

        if self.verbose:
            print(
                "[TorchDR] Affinity : sparsity mode enabled, computing "
                f"{k} nearest neighbors."
            )
        C_, temp_indices = self._distance_matrix(X, k=k)

        # Compute rho as the average distance between the 4th to 6th neighbors
        sq_neighbor_distances, _ = kmin(C_, k=6, dim=1)
        self.rho_ = torch.sqrt(sq_neighbor_distances)[:, 3:6].mean(dim=1).contiguous()

        rho_i = self.rho_.unsqueeze(1)  # Shape: (n_samples, 1)
        rho_j = self.rho_[temp_indices]  # Shape: (n_samples, k)
        normalized_C = C_ / rho_i * rho_j

        # Compute final NN indices
        _, local_indices = kmin(normalized_C, k=self.n_neighbors, dim=1)
        final_indices = torch.gather(temp_indices, 1, local_indices.to(torch.int64))

        return None, final_indices  # PACMAP only uses the NN indices
