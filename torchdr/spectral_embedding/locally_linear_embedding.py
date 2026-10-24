"""Standard Locally Linear Embedding with a dense PyTorch eigensolver."""

from typing import Optional

import torch

from torchdr.base import DRModule
from torchdr.distance import pairwise_distances
from torchdr.utils import svd_flip


class LocallyLinearEmbedding(DRModule):
    r"""Preserve local barycentric reconstruction weights in an embedding [1]_.

    For each sample, find Euclidean nearest neighbors and solve a regularized
    least-squares reconstruction constrained to weights summing to one. The
    embedding uses the smallest nontrivial eigenvectors of
    :math:`(I-W)^T(I-W)`, where ``W`` contains those weights.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of non-self neighbors, less than the number of unique samples.
    n_components : int, default=2
        Embedding dimension, less than the number of unique samples.
    reg : float, default=1e-3
        Nonnegative regularization multiplied by the local covariance trace.
        A zero trace uses ``reg`` itself, matching sklearn's standard method.
    device : str, default="auto"
        Device for neighbor search, local solves and the eigendecomposition.
    verbose : bool, default=False
        Whether to log information.
    random_state : int or None, default=None
        Random seed for the base estimator. The dense solver itself does not
        use randomness; repeated eigenvalues can still have nonunique bases.
    process_duplicates : bool, default=True
        Fit unique samples and expand their embedding to the original rows.
    **kwargs : dict
        Additional arguments to :class:`~torchdr.DRModule`.

    Attributes
    ----------
    neighbors_ : torch.Tensor of shape (n_unique_samples, n_neighbors)
        Neighbor indices in the fitted (possibly deduplicated) sample space.
    weights_ : torch.Tensor of shape (n_unique_samples, n_neighbors)
        Barycentric weights for each local reconstruction.
    reconstruction_error_ : float
        Sum of the selected eigenvalues, before any duplicate expansion.
    embedding_ : torch.Tensor of shape (n_samples, n_components)
        Low-dimensional embedding.

    Notes
    -----
    Implements only standard LLE, not Hessian, modified or LTSA variants.
    This implementation stores dense quadratic matrices and uses a dense
    eigensolver. Local solves and the eigendecomposition use float64 to
    resolve the smallest eigenvalues; embeddings retain the input dtype.
    Lazy/FAISS backends and out-of-sample transforms are not
    supported. It is intended for small datasets.

    References
    ----------
    .. [1] Roweis, S. and Saul, L. (2000). Nonlinear Dimensionality Reduction
       by Locally Linear Embedding. Science, 290(5500), 2323-2326.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_components: int = 2,
        reg: float = 1e-3,
        device: str = "auto",
        verbose: bool = False,
        random_state: Optional[int] = None,
        process_duplicates: bool = True,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            verbose=verbose,
            random_state=random_state,
            process_duplicates=process_duplicates,
            **kwargs,
        )
        self.n_neighbors = n_neighbors
        self.reg = reg
        if self.backend is not None:
            raise NotImplementedError(
                "LocallyLinearEmbedding requires a dense backend."
            )

    def _fit_transform(self, X, y=None):
        """Solve the local reconstruction problems and the global embedding."""
        n = X.shape[0]
        for name in ("n_neighbors", "n_components"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value < n
            ):
                raise ValueError(f"{name} must be an integer in [1, n_samples - 1].")
        if not isinstance(self.reg, (int, float)) or not 0 <= self.reg < float("inf"):
            raise ValueError("reg must be finite and nonnegative.")
        if self.device != "auto":
            X = X.to(self.device)
        output_dtype = X.dtype
        X = X.to(torch.float64)
        _, neighbors = pairwise_distances(
            X, k=self.n_neighbors, return_indices=True, exclude_diag=True
        )
        differences = X[neighbors] - X[:, None, :]
        covariance = differences @ differences.transpose(-1, -2)
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        ridge = self.reg * torch.where(trace > 0, trace, torch.ones_like(trace))
        covariance = covariance + ridge[:, None, None] * torch.eye(
            self.n_neighbors, dtype=X.dtype, device=X.device
        )
        weights = torch.linalg.solve(
            covariance,
            torch.ones(n, self.n_neighbors, 1, dtype=X.dtype, device=X.device),
        ).squeeze(-1)
        weights = weights / weights.sum(-1, keepdim=True)
        W = torch.zeros(n, n, dtype=X.dtype, device=X.device).scatter(
            1, neighbors, weights
        )
        residual = torch.eye(n, dtype=X.dtype, device=X.device) - W
        eigenvalues, eigenvectors = torch.linalg.eigh(residual.T @ residual)
        embedding = eigenvectors[:, 1 : self.n_components + 1]
        embedding, _ = svd_flip(embedding, torch.zeros_like(embedding).T)
        self.neighbors_ = neighbors
        self.weights_ = weights
        self.reconstruction_error_ = eigenvalues[1 : self.n_components + 1].sum().item()
        self.embedding_ = embedding.to(output_dtype)
        return self.embedding_
