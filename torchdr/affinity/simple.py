# -*- coding: utf-8 -*-
"""
Common simple affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import numpy as np
from typing import Tuple

from torchdr.utils import normalize_matrix
from torchdr.affinity.base import Affinity, LogAffinity


class ScalarProductAffinity(Affinity):
    r"""
    Computes the scalar product affinity matrix :math:`\mathbf{X} \mathbf{X}^T`
    where :math:`\mathbf{X}` is the input data.
    """

    def __init__(
        self,
        device: str = "cuda",
        keops: bool = True,
        verbose: bool = True,
        centering: bool = False,
    ):
        super().__init__(metric="angular", device=device, keops=keops, verbose=verbose)
        self.centering = centering

    def fit(self, X: torch.Tensor | np.ndarray):
        super().fit(X)
        if self.centering:
            self.data_ = self.data_ - self.data_.mean(0)
        self.affinity_matrix_ = -self._pairwise_distance_matrix(self.data_)


class GibbsAffinity(LogAffinity):
    r"""
    Computes the Gibbs affinity matrix :math:`\exp( - \mathbf{C} / \sigma)`
    where :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma` is the bandwidth parameter.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth parameter.
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix.
    metric : str, optional
        Metric to use for pairwise distances computation.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        normalization_dim: int | Tuple[int] = (0, 1),
        metric: str = "euclidean",
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.sigma = sigma
        self.normalization_dim = normalization_dim

    def fit(self, X: torch.Tensor | np.ndarray):
        super().fit(X)
        C = self._pairwise_distance_matrix(self.data_)
        log_P = -C / self.sigma
        self.log_affinity_matrix_ = normalize_matrix(
            log_P, dim=self.normalization_dim, log=True
        )


class StudentAffinity(LogAffinity):
    def __init__(
        self,
        degrees_of_freedom: int = 1,
        normalization_dim: int | Tuple[int] = (0, 1),
        metric: str = "euclidean",
        device: str = None,
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)
        self.normalization_dim = normalization_dim
        self.degrees_of_freedom = degrees_of_freedom

    def fit(self, X: torch.Tensor | np.ndarray):
        super().fit(X)
        C = self._pairwise_distance_matrix(self.data_)
        C /= self.degrees_of_freedom
        C += 1.0
        log_P = -0.5 * (self.degrees_of_freedom + 1) * C.log()
        self.log_affinity_matrix_ = normalize_matrix(
            log_P, dim=self.normalization_dim, log=True
        )
