# -*- coding: utf-8 -*-
"""
Common (simple) affinity matrices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from abc import ABC, abstractmethod
from typing import Union, Tuple

import torch
import numpy as np
from torchdr.utils import pairwise_distances, normalize_matrix, to_torch


class Affinity(ABC):
    """
    Base class for affinity matrices.
    """

    def __init__(
        self,
        metric: str = "euclidean",
        device: str = "cuda",
        keops: bool = True,
        verbose: bool = True,
    ):
        self.log = {}
        self.metric = metric
        self.device = device
        self.keops = keops
        self.verbose = verbose

    @abstractmethod
    def fit(self, X: torch.Tensor | np.ndarray):
        self.data_ = to_torch(X, device=self.device, verbose=self.verbose)
        return self

    def fit_transform(self, X: torch.Tensor | np.ndarray):
        """
        Computes the affinity matrix from input data X.
        """
        self.fit(X)
        self._check_is_fitted(
            msg="[TorchDR] Error : affinity_matrix_ should be computed in fit method."
        )
        return self.affinity_matrix_  # type: ignore

    def _ground_cost_matrix(self, X: torch.Tensor):
        return pairwise_distances(X, metric=self.metric, keops=self.keops)

    def _check_is_fitted(self, msg: str = None):
        assert hasattr(self, "affinity_matrix_"), (
            msg or "[TorchDR] Error : Affinity not fitted."
        )

    def get_batch(self, indices: torch.Tensor):
        self._check_is_fitted()
        assert (
            indices.dim() == 2
        ), '[TorchDR] Error : indices in "get_batch" should be a 2D torch tensor '
        "of shape (n_batch, batch_size)."
        return self


class ScalarProductAffinity(Affinity):
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
        self.affinity_matrix_ = -self._ground_cost_matrix(self.data_)


class LogAffinity(Affinity):
    """Computes an affinity matrix from an affinity matrix in log space."""

    def __init__(
        self,
        metric: str = "euclidean",
        device: str = "cuda",
        keops: bool = True,
        verbose: bool = True,
    ):
        super().__init__(metric=metric, device=device, keops=keops, verbose=verbose)

    def fit_transform(self, X: torch.Tensor | np.ndarray, log: bool = False):
        self.fit(X)
        assert hasattr(
            self, "log_affinity_matrix_"
        ), "[TorchDR] Affinity (Error) : log_affinity_matrix_ should be computed "
        "in  fit method of a LogAffinity."

        if log:  # return the log of the affinity matrix
            return self.log_affinity_matrix_  # type: ignore
        else:
            if not hasattr(self, "affinity_matrix_"):
                self.affinity_matrix_ = self.log_affinity_matrix_.exp()  # type: ignore
            return self.affinity_matrix_


class GibbsAffinity(LogAffinity):
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
        C = self._ground_cost_matrix(self.data_)
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
        C = self._ground_cost_matrix(self.data_)
        C /= self.degrees_of_freedom
        C += 1.0
        log_P = -0.5 * (self.degrees_of_freedom + 1) * C.log()
        self.log_affinity_matrix_ = normalize_matrix(
            log_P, dim=self.normalization_dim, log=True
        )
