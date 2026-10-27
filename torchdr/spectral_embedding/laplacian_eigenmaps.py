"""Laplacian Eigenmaps using a dense normalized graph Laplacian."""

import warnings
from numbers import Integral
from typing import Optional

import torch

from torchdr.affinity import Affinity, NormalizedGaussianAffinity
from torchdr.base import DRModule
from torchdr.utils import svd_flip


class LaplacianEigenmaps(DRModule):
    r"""Embed data using the eigenvectors of a normalized graph Laplacian.

    Solves :math:`(D-W)y = \lambda D y` [1]_, where ``W`` is the symmetric
    affinity and ``D`` its degree matrix. The first, constant eigenvector is
    discarded. Like ``sklearn.manifold.SpectralEmbedding(affinity="rbf")``,
    the default uses a dense Gaussian graph, not a nearest-neighbor graph.

    Parameters
    ----------
    affinity : Affinity or None, default=None
        Dense affinity. Defaults to an unnormalized Gaussian affinity with
        bandwidth 1. Asymmetric affinities are averaged with their transpose;
        self-edges are ignored.
    n_components : int, default=2
        Embedding dimension, strictly less than the number of unique samples.
    device : str, default="auto"
        Device for the default Gaussian affinity. A custom affinity controls
        its own device.
    verbose : bool, default=False
        Whether to log information.
    random_state : int or None, default=None
        Random seed. The dense eigensolver itself is deterministic up to
        eigenvector signs and rotations in repeated eigenspaces.
    process_duplicates : bool, default=True
        Fit unique samples and expand their embeddings back to the original rows.
    **kwargs : dict
        Additional arguments to :class:`~torchdr.DRModule`.

    Attributes
    ----------
    eigenvalues_ : torch.Tensor of shape (n_components,)
        Smallest nontrivial normalized-Laplacian eigenvalues.
    embedding_ : torch.Tensor of shape (n_samples, n_components)
        Degree-normalized eigenvectors.

    Notes
    -----
    This implementation uses quadratic storage and a dense eigensolver. KeOps,
    FAISS and sparse affinity outputs are not supported. There is no
    out-of-sample transform. Disconnected graphs issue a warning because their
    null space contains more than the constant mode.

    References
    ----------
    .. [1] Belkin, M. and Niyogi, P. (2003). Laplacian Eigenmaps for
       Dimensionality Reduction and Data Representation. Neural Computation,
       15(6), 1373-1396.
    """

    def __init__(
        self,
        affinity: Optional[Affinity] = None,
        n_components: int = 2,
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
        if self.backend is not None:
            raise NotImplementedError("LaplacianEigenmaps requires a dense backend.")
        self.affinity = affinity
        if affinity is not None and affinity.backend is not None:
            raise NotImplementedError("LaplacianEigenmaps requires a dense affinity.")

    def _fit_transform(self, X, y=None):
        """Compute the nontrivial generalized Laplacian eigenvectors."""
        n = X.shape[0]
        if (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, Integral)
            or not 1 <= self.n_components < n
        ):
            raise ValueError("n_components must be an integer in [1, n_samples - 1].")
        affinity = self.affinity
        if affinity is None:
            affinity = NormalizedGaussianAffinity(
                normalization_dim=None, device=self.device
            )
        W = affinity(X)
        if isinstance(W, tuple) and len(W) == 2 and W[1] is None:
            W = W[0]
        if not isinstance(W, torch.Tensor) or W.layout != torch.strided:
            raise NotImplementedError("LaplacianEigenmaps requires a dense affinity.")
        if W.shape != (n, n) or not torch.isfinite(W).all() or (W < 0).any():
            raise ValueError("Affinity must be a finite nonnegative square matrix.")
        W = (W + W.T) / 2
        W = W - torch.diag_embed(W.diagonal())
        degree = W.sum(1)
        # Isolated vertices use unit scaling, matching the normalized Laplacian.
        scale = torch.where(degree > 0, degree.sqrt(), torch.ones_like(degree))
        L = -W / scale[:, None] / scale[None, :]
        L = L + torch.diag_embed((degree > 0).to(W.dtype))
        values, vectors = torch.linalg.eigh(L)
        tol = torch.finfo(values.dtype).eps * n * 10
        if (values.abs() <= tol).sum() > 1:
            warnings.warn(
                "Graph is not fully connected; the embedding may not be meaningful.",
                UserWarning,
                stacklevel=2,
            )
        vectors = vectors[:, 1 : self.n_components + 1] / scale[:, None]
        vectors, _ = svd_flip(vectors, torch.zeros_like(vectors).T)
        self.eigenvalues_ = values[1 : self.n_components + 1]
        self.embedding_ = vectors
        return self.embedding_
