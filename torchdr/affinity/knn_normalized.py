"""Affinity matrices with normalizations using nearest neighbor distances."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Cédric Vincent-Cuaz <cedric.vincent-cuaz@inria.fr>
#
# License: BSD 3-Clause License

from typing import Tuple, Union, Optional

import torch

from torchdr.affinity.base import Affinity, LogAffinity
from torchdr.utils import (
    batch_transpose,
    kmin,
    logsumexp_red,
    sum_red,
    wrap_vectors,
    pairwise_distances,
    LazyTensorType,
)
from torchdr.utils import (
    identity_matrix,
    diffusion_from_affinity,
    apply_anisotropy,
    matrix_power,
)


@wrap_vectors
def _log_SelfTuning(C, sigma):
    r"""Return the self-tuning affinity matrix with sample-wise bandwidth.

    The bandwidth is determined by the distance from a point
    to its K-th neirest neighbor in log domain.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        Pairwise distance matrix.
    sigma : torch.Tensor of shape (n,)
        Sample-wise bandwidth parameter.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor
    """
    sigma_t = batch_transpose(sigma)
    return -C / (sigma * sigma_t)


@wrap_vectors
def _log_MAGIC(C, sigma):
    r"""Return the MAGIC affinity matrix with sample-wise bandwidth.

    The bandwidth is determined by the distance from a point
    to its K-th neirest neighbor in log domain.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        Pairwise distance matrix.
    sigma : torch.Tensor of shape (n,)
        Sample-wise bandwidth parameter.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor
    """
    return -C / sigma


@wrap_vectors
def _log_AlphaDecay(C, sigma, alpha):
    r"""Return the alpha-decay affinity matrix with sample-wise bandwidth.

    The bandwidth is determined by the distance from a point
    to its K-th neirest neighbor in log domain.

    Parameters
    ----------
    C : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
        Pairwise distance matrix.
    sigma : torch.Tensor of shape (n,)
        Sample-wise bandwidth parameter.
    alpha : float, optional
        Exponent for the alpha-decay kernel.

    Returns
    -------
    log_P : torch.Tensor or pykeops.torch.LazyTensor
    """
    return -(C / sigma).pow(alpha)


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
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.K = K
        self.normalization_dim = normalization_dim

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

        minK_values, minK_indices = kmin(C, k=self.K, dim=1)
        self.sigma_ = minK_values[:, -1]
        log_affinity_matrix = _log_SelfTuning(C, self.sigma_)

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                log_affinity_matrix, self.normalization_dim
            )
            log_affinity_matrix = log_affinity_matrix - self.log_normalization_

        return log_affinity_matrix


class MAGICAffinity(Affinity):
    r"""Compute the MAGIC affinity introduced in :cite:`van2018recovering`.

    The construction is as follows. First, it computes a Gaussian
    kernel with sample-wise bandwidth :math:`\mathbf{\sigma} \in \mathbb{R}^n`.

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
        K-th neirest neighbor .
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
    """

    def __init__(
        self,
        K: int = 7,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.K = K

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

        minK_values, minK_indices = kmin(C, k=self.K, dim=1)
        self.sigma_ = minK_values[:, -1]
        affinity_matrix = _log_MAGIC(C, self.sigma_).exp()
        affinity_matrix = (affinity_matrix + batch_transpose(affinity_matrix)) / 2

        self.normalization_ = sum_red(affinity_matrix, 1)
        affinity_matrix = affinity_matrix / self.normalization_

        return affinity_matrix


class AlphaDecayAffinity(Affinity):
    r"""Compute the alpha-decay affinity introduced in :cite:`moon2019visualizing`.

    The affinity has a sample-wise bandwidth :math:`\mathbf{\sigma} \in \mathbb{R}^n`.

    .. math::
        P_{ij} \leftarrow \exp \left( - \left( \frac{C_{ij}}{\sigma_i} \right)^\alpha \right)

    In the above, :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma_i` is the distance from the K'th nearest neighbor of data point
    :math:`\mathbf{x}_i`.

    Parameters
    ----------
    K : int, optional
        K-th neirest neighbor .
    alpha : float, optional
        Exponent for the alpha-decay kernel.
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
    """

    def __init__(
        self,
        K: int = 7,
        alpha: float = 2.0,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = None,
        backend: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.K = K
        self.alpha = alpha

    def _compute_affinity(self, X: torch.Tensor):
        C, _ = self._distance_matrix(X)
        minK_values, _ = kmin(C, k=self.K, dim=1)
        self.sigma_ = minK_values[:, -1]
        affinity_matrix = _log_AlphaDecay(C, self.sigma_, self.alpha).exp()
        affinity_matrix = (affinity_matrix + batch_transpose(affinity_matrix)) / 2

        self.normalization_ = sum_red(affinity_matrix, 1)
        affinity_matrix = affinity_matrix / self.normalization_

        return affinity_matrix


class NegPotentialAffinity(Affinity):
    def __init__(
        self,
        metric: str = "sqeuclidean",
        device: str = None,
        backend: Optional[str] = None,
        verbose: bool = False,
        sigma: float = 2.0,
        anisotropy: float = 0.0,
        K: int = 7,
        alpha: float = 2.0,
        t: int = 5,
        eps: float = 1e-5,
    ):
        super().__init__(
            metric=metric,
            device=device,
            backend=backend,
            verbose=verbose,
            zero_diag=False,
        )
        self.base_affinity = AlphaDecayAffinity(
            K=K,
            alpha=alpha,
            metric=metric,
            device=device,
            backend=backend,
            verbose=verbose,
        )
        self.sigma = sigma
        self.anisotropy = anisotropy
        self.t = t
        self.eps = eps
        self.keops = backend == "keops"

    @staticmethod
    def potential_dist(
        affinity: LazyTensorType, eps: float = 1e-5, backend: Optional[str] = None
    ) -> LazyTensorType:
        r"""Compute the potential distance matrix from the affinity matrix.

        Parameters
        ----------
        affinity : torch.Tensor or pykeops.torch.LazyTensor of shape (n, n)
            Affinity matrix.
        eps : float, optional
            Small value to avoid numerical issues.
        backend : {"keops", "faiss", None}, optional
            Which backend to use for handling sparsity and memory efficiency.
            Default is None.

        Returns
        -------
        potential_dist : torch.Tensor or pykeops.torch.LazyTensor
        """
        log_affinity = -(affinity + eps).log()
        potential_dist = pairwise_distances(
            log_affinity, metric="euclidean", backend=backend
        )
        return potential_dist

    def _compute_affinity(self, X: torch.Tensor):
        affinity = self.base_affinity(X)
        affinity = apply_anisotropy(affinity, self.anisotropy)
        diffusion = diffusion_from_affinity(affinity)
        diffusion = matrix_power(diffusion, self.t, self.keops)
        dist = self.potential_dist(diffusion)
        # symetrize
        dist = (dist + batch_transpose(dist)) / 2
        # zero the diagonal
        identity = identity_matrix(dist.shape[-1], self.keops, X.device, X.dtype)
        dist = dist - identity * dist.diag()
        return -1.0 * dist
