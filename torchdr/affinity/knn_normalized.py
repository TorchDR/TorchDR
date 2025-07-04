"""Affinity matrices with normalizations using nearest neighbor distances."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Cédric Vincent-Cuaz <cedric.vincent-cuaz@inria.fr>
#
# License: BSD 3-Clause License

from typing import Tuple, Union, Optional

import torch

from torchdr.affinity.base import Affinity, LogAffinity, SparseAffinity
from torchdr.utils import (
    matrix_transpose,
    kmin,
    logsumexp_red,
    sum_red,
    wrap_vectors,
    matrix_power,
    check_neighbor_param,
    binary_search,
    compile_if_requested,
)
from torchdr.utils.sparse import sym_sparse_op
from torchdr.distance import pairwise_distances


@wrap_vectors
def _log_P_SelfTuning(C, sigma):
    sigma_t = matrix_transpose(sigma)
    return -C / (sigma * sigma_t)


@wrap_vectors
def _log_P_MAGIC(C, sigma):
    return -C / sigma


@wrap_vectors
def _log_P_UMAP(C, rho, sigma):
    return -(C - rho) / sigma


@wrap_vectors
def _log_P_PHATE(C, sigma, alpha=10.0):
    return -((C / sigma) ** alpha)


class SelfTuningAffinity(LogAffinity):
    r"""Self-tuning affinity introduced in :cite:`zelnik2004self`.

    The affinity has a sample-wise bandwidth :math:`\mathbf{\sigma} \in \mathbb{R}^n`.

    .. math::
        \exp \left( - \frac{C_{ij}}{\sigma_i \sigma_j} \right)

    In the above, :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma_i` is the distance from the K'th nearest neighbor of data point
    :math:`\mathbf{x}_i`.

    Parameters
    ----------
    K : int, optional
        K-th neirest neighbor .
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix.
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
    compile : bool, optional
        Whether to compile the computation. Default is False.
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """

    def __init__(
        self,
        K: int = 7,
        normalization_dim: Union[int, Tuple[int]] = (0, 1),
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            compile=compile,
            _pre_processed=_pre_processed,
        )
        self.K = K
        self.normalization_dim = normalization_dim

    @compile_if_requested
    def _compute_log_affinity(self, X: torch.Tensor):
        r"""Fit the self-tuning affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor
            Input data.

        Returns
        -------
        log_affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix in log domain.
        """
        C, _ = self._distance_matrix(X)

        minK_values, _ = kmin(C, k=self.K, dim=1)
        self.sigma_ = minK_values[:, -1]
        log_affinity_matrix = _log_P_SelfTuning(C, self.sigma_)

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                log_affinity_matrix, self.normalization_dim
            )
            log_affinity_matrix = log_affinity_matrix - self.log_normalization_

        return log_affinity_matrix


class MAGICAffinity(Affinity):
    r"""Compute the MAGIC affinity with alpha-decay kernel introduced in :cite:`van2018recovering`.

    The construction is as follows. First, it computes a generalized
    kernel with sample-wise bandwidth :math:`\mathbf{\sigma} \in \mathbb{R}^n`:

    .. math::
        P_{ij} \leftarrow \exp \left( - \frac{C_{ij}}{\sigma_i} \right)

    In the above, :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma_i` is the distance from the K'th nearest neighbor of data point
    :math:`\mathbf{x}_i`.

    Then it averages the affinity matrix with its transpose:

    .. math::
        P_{ij} \leftarrow \frac{P_{ij} + P_{ji}}{2} \:.

    Finally, it normalizes the affinity matrix along each row:

    .. math::
        P_{ij} \leftarrow \frac{P_{ij}}{\sum_{t} P_{it}} \:.


    Parameters
    ----------
    K : int, optional
        K-th neirest neighbor. Default is 7.
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
    compile : bool, optional
        Whether to compile the computation. Default is False.
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """

    def __init__(
        self,
        K: int = 7,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            compile=compile,
            _pre_processed=_pre_processed,
        )
        self.K = K

    @compile_if_requested
    def _compute_affinity(self, X: torch.Tensor):
        r"""Fit the MAGIC affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor
            Input data.

        Returns
        -------
        affinity_matrix : torch.Tensor or pykeops.torch.LazyTensor
            The computed affinity matrix.
        """
        C, _ = self._distance_matrix(X)

        minK_values, _ = kmin(C, k=self.K, dim=1)
        self.sigma_ = minK_values[:, -1]
        affinity_matrix = _log_P_MAGIC(C, self.sigma_).exp()
        affinity_matrix = (affinity_matrix + matrix_transpose(affinity_matrix)) / 2
        affinity_matrix = affinity_matrix / sum_red(affinity_matrix, dim=1)

        return affinity_matrix


class PHATEAffinity(Affinity):
    r"""Compute the potential affinity used in PHATE :cite:`moon2019visualizing`.

    The method follows these steps:
    1. Compute pairwise distance matrix
    2. Find k-th nearest neighbor distances to set bandwidth sigma
    3. Compute base affinity with alpha-decay kernel: exp(-((d/sigma)^alpha))
    4. Symmetrize the affinity matrix
    5. Row-normalize to create diffusion matrix
    6. Raise diffusion matrix to power t (diffusion steps)
    7. Compute potential distances from the diffused matrix
    8. Return negative potential distances as affinities

    Parameters
    ----------
    metric : str, optional (default="euclidean")
        Metric to use for pairwise distances computation.
    device : str, optional (default=None)
        Device to use for computations. If None, uses the device of input data.
    backend : {"keops", "faiss", None}, optional (default=None)
        Which backend to use for handling sparsity and memory efficiency.
    verbose : bool, optional (default=False)
        Whether to print verbose output during computation.
    k : int, optional (default=5)
        Number of nearest neighbors used to determine bandwidth parameter sigma.
    alpha : float, optional (default=10.0)
        Exponent for the alpha-decay kernel in affinity computation.
    t : int, optional (default=5)
        Number of diffusion steps (power to raise diffusion matrix).
    compile : bool, optional
        Whether to compile the computation. Default is False.
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """

    def __init__(
        self,
        metric: str = "euclidean",
        device: str = None,
        backend: Optional[str] = None,
        verbose: bool = False,
        k: int = 5,
        alpha: float = 10.0,
        t: int = 5,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        if backend == "faiss" or backend == "keops":
            raise ValueError(
                f"[TorchDR] ERROR : {self.__class__.__name__} class does not support backend {backend}."
            )

        super().__init__(
            metric=metric,
            device=device,
            backend=backend,
            verbose=verbose,
            compile=compile,
            zero_diag=False,
            _pre_processed=_pre_processed,
        )

        self.k = k
        self.alpha = alpha
        self.t = t

    @compile_if_requested
    def _compute_affinity(self, X: torch.Tensor):
        C, _ = self._distance_matrix(X)

        minK_values, _ = kmin(C, k=self.k, dim=1)
        self.sigma_ = minK_values[:, -1]
        affinity = _log_P_PHATE(C, self.sigma_, self.alpha).exp()
        affinity = (affinity + matrix_transpose(affinity)) / 2
        affinity = affinity / sum_red(affinity, dim=1)
        affinity = matrix_power(affinity, self.t)
        affinity = -pairwise_distances(
            -affinity.clamp(min=1e-12).log(), metric="euclidean", backend=self.backend
        )[0]
        return affinity


class UMAPAffinity(SparseAffinity):
    r"""Compute the input affinity used in UMAP :cite:`mcinnes2018umap`.

    The algorithm computes via root search the variable
    :math:`\mathbf{\sigma}^* \in \mathbb{R}^n_{>0}` such that

    .. math::
        \forall (i,j), \: P_{ij} = \exp(- (C_{ij} - \rho_i) / \sigma^\star_i) \quad \text{where} \quad \forall i, \: \sum_j P_{ij} = \log (\mathrm{n_neighbors})

    and :math:`\rho_i = \min_j C_{ij}`.

    Parameters
    ----------
    n_neighbors : float, optional
        Number of effective nearest neighbors to consider. Similar to the perplexity.
    max_iter : int, optional
        Maximum number of iterations for the root search.
    sparsity : bool, optional
        Whether to use sparsity mode.
        Default is True.
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
    compile : bool, optional
        Whether to compile the computation. Default is False.
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 30,
        max_iter: int = 1000,
        sparsity: bool = True,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter

        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=sparsity,
            compile=compile,
            _pre_processed=_pre_processed,
        )

    @compile_if_requested
    def _compute_sparse_log_affinity(self, X: torch.Tensor):
        r"""Compute the input affinity matrix of UMAP from input data X.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which affinity is computed.

        Returns
        -------
        self : UMAPAffinityIn
            The fitted instance.
        """

        n_samples_in = X.shape[0]
        n_neighbors = check_neighbor_param(self.n_neighbors, n_samples_in)

        if self.sparsity:
            if self.verbose:
                self.logger.info(
                    f"Sparsity mode enabled, computing {n_neighbors} nearest neighbors."
                )
            # when using sparsity, we construct a reduced distance matrix
            # of shape (n_samples, n_neighbors)
            C_, indices = self._distance_matrix(X, k=n_neighbors)
        else:
            C_, indices = self._distance_matrix(X)

        self.rho_ = kmin(C_, k=1, dim=1)[0].squeeze().contiguous()

        log_n_neighbors = torch.log2(
            torch.tensor(n_neighbors, dtype=X.dtype, device=X.device)
        )

        def marginal_gap(eps):
            log_marg = _log_P_UMAP(C_, self.rho_, eps).logsumexp(1)
            return log_marg.exp().squeeze() - log_n_neighbors

        self.eps_ = binary_search(
            f=marginal_gap,
            n=n_samples_in,
            max_iter=self.max_iter,
            dtype=X.dtype,
            device=X.device,
        )

        # Compute log affinity matrix
        log_affinity_matrix = _log_P_UMAP(C_, self.rho_, self.eps_)

        # symmetrize
        affinity_matrix = log_affinity_matrix.exp()
        out_val, out_idx = sym_sparse_op(affinity_matrix, indices)

        return out_val, out_idx


class PACMAPAffinity(SparseAffinity):
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
    compile : bool, optional
        Whether to compile the computation. Default is False.
    _pre_processed : bool, optional
        If True, assumes inputs are already torch tensors on the correct device
        and skips the `to_torch` conversion. Default is False.
    """  # noqa: E501

    def __init__(
        self,
        n_neighbors: float = 10,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        self.n_neighbors = n_neighbors

        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
            sparsity=True,  # PACMAP uses sparsity mode
            compile=compile,
            _pre_processed=_pre_processed,
        )

    @compile_if_requested
    def _compute_sparse_log_affinity(self, X: torch.Tensor):
        r"""Compute the input affinity matrix of PACMAP from input data X.

        Parameters
        ----------
        X : torch.Tensor
            Input data.

        Returns
        -------
        self : PACMAPAffinityIn
            The fitted instance.
        """
        n_samples_in = X.shape[0]
        k = min(self.n_neighbors + 50, n_samples_in)
        k = check_neighbor_param(k, n_samples_in)

        if self.verbose:
            self.logger.info(f"Sparsity mode enabled, computing {k} nearest neighbors.")

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
