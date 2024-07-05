# -*- coding: utf-8 -*-
"""
Affinity matrices with normalizations using nearest neighbor distances
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from typing import Tuple

from torchdr.utils import logsumexp_red
from torchdr.affinity.base import LogAffinity
from torchdr.utils import (
    extract_batch_normalization,
    kmin,
    wrap_vectors,
    batch_transpose,
)


@wrap_vectors
def _log_SelfTuningGibbs(C, sigma):
    r"""
    Returns the self-tuning Gibbs affinity matrix with sample-wise bandwidth
    determined by the distance from a point to its K-th neirest neighbor
    in log domain.

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


class SelfTuningGibbsAffinity(LogAffinity):
    r"""
    Computes the self-tuning [22]_ Gibbs affinity matrix with
    sample-wise bandwidth :math:`\mathbf{\sigma} \in \mathbb{R}^n`.

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
    nodiag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity.

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
        nodiag: bool = True,
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric, nodiag=nodiag, device=device, keops=keops, verbose=verbose
        )
        self.K = K
        self.normalization_dim = normalization_dim

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the local Gibbs affinity model to the provided data.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data.

        Returns
        -------
        self : LocalGibbsAffinity
            The fitted local Gibbs affinity model.
        """
        super().fit(X)
        C = self._pairwise_distance_matrix(self.data_)

        minK_values, minK_indices = kmin(C, k=self.K, dim=1)
        self.sigma_ = minK_values[:, -1]
        self.log_affinity_matrix_ = _log_SelfTuningGibbs(C, self.sigma_)

        if self.normalization_dim is not None:
            self.log_normalization_ = logsumexp_red(
                self.log_affinity_matrix_, self.normalization_dim
            )
            self.log_affinity_matrix_ = (
                self.log_affinity_matrix_ - self.log_normalization_
            )

        return self

    def get_batch(self, indices: torch.Tensor, log: bool = False):
        r"""
        Extracts the affinity submatrix corresponding to the indices.

        Parameters
        ----------
        indices : torch.Tensor of shape (n_batch, batch_size)
            Indices of the batch.
        log : bool, optional
            If True, returns the log of the affinity matrix.

        Returns
        -------
        P_batch : torch.Tensor or pykeops.torch.LazyTensor
            of shape (n_batch, batch_size, batch_size)
            The affinity matrix for the batch indices.
            In log domain if `log` is True.
        """
        C_batch = super().get_batch(indices)
        sigma_batch = self.sigma_[indices]
        log_P_batch = _log_SelfTuningGibbs(C_batch, sigma_batch)

        if self.normalization_dim is not None:
            log_normalization_batch = extract_batch_normalization(
                self.log_normalization_, indices, self.normalization_dim
            )
            log_P_batch = log_P_batch - log_normalization_batch

        if log:
            return log_P_batch
        else:
            return log_P_batch.exp()
