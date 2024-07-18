# -*- coding: utf-8 -*-
"""Affinity matrices with normalizations using nearest neighbor distances."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Cédric Vincent-Cuaz <cedric.vincent-cuaz@inria.fr>
#
# License: BSD 3-Clause License

import torch
from typing import Tuple

from torchdr.affinity.base import Affinity, LogAffinity
from torchdr.utils import (
    kmin,
    wrap_vectors,
    batch_transpose,
    logsumexp_red,
    sum_red,
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


class SelfTuningAffinity(LogAffinity):
    r"""Compute the self-tuning affinity introduced in [22]_.

    The affinity has a sample-wise bandwidth :math:`\mathbf{\sigma} \in \mathbb{R}^n`.

    .. math::
        \exp \left( - \frac{C_{ij}}{\sigma_i \sigma_j} \right)

    In the above, :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma_i` is the distance from the K’th nearest neighbor of data point
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
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity. Default is False.

    References
    ----------
    .. [22] Max Zelnik-Manor, L., & Perona, P. (2004).
            Self-tuning spectral clustering.
            Advances in neural information processing systems (NeurIPS).
    """

    def __init__(
        self,
        K: int = 7,
        normalization_dim: int | Tuple[int] = (0, 1),
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = None,
        keops: bool = True,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
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
        C = self._distance_matrix(X)

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
    r"""Compute the MAGIC affinity introduced in [23]_.

    The construction is as follows. First, it computes a Gaussian
    kernel with sample-wise bandwidth :math:`\mathbf{\sigma} \in \mathbb{R}^n`.

    .. math::
        P_{ij} \leftarrow \exp \left( - \frac{C_{ij}}{\sigma_i} \right)

    In the above, :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma_i` is the distance from the K’th nearest neighbor of data point
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
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity. Default is False.

    References
    ----------
    .. [23] Van Dijk, D., Sharma, R., Nainys, J., Yim, K., Kathail, P., Carr, A.
            J., ... & Pe’er, D. (2018).
            Recovering Gene Interactions from Single-Cell Data Using Data Diffusion
            Cell, 174(3).
    """

    def __init__(
        self,
        K: int = 7,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = None,
        keops: bool = True,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
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
        C = self._distance_matrix(X)

        minK_values, minK_indices = kmin(C, k=self.K, dim=1)
        self.sigma_ = minK_values[:, -1]
        affinity_matrix = _log_MAGIC(C, self.sigma_).exp()
        affinity_matrix = (affinity_matrix + batch_transpose(affinity_matrix)) / 2

        self.normalization_ = sum_red(affinity_matrix, 1)
        affinity_matrix = affinity_matrix / self.normalization_

        return affinity_matrix
