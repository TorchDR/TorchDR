# -*- coding: utf-8 -*-
"""
Spectral methods for dimensionality reduction
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from torchdr.base import DRModule
from torchdr.utils import (
    svd_flip,
    handle_backend,
)


class PCA(DRModule):
    def __init__(self, n_components=2, device=None, verbose=False):
        super().__init__(n_components=n_components, device=device, verbose=verbose)

    def fit(self, X):
        super().fit(X)
        self.mean_ = self.X_.mean(0, keepdim=True)
        U, _, V = torch.linalg.svd(self.X_ - self.mean_, full_matrices=False)
        U, V = svd_flip(U, V)  # flip eigenvectors' sign to enforce deterministic output
        self.components_ = V[: self.n_components]
        return self

    @handle_backend
    def transform(self, X):
        return (X - self.mean_) @ self.components_.T
