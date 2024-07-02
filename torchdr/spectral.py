# -*- coding: utf-8 -*-
"""
Spectral methods for dimensionality reduction
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from scipy.sparse.linalg import aslinearoperator, eigsh

from torchdr.base import DRModule
from torchdr.utils import (
    svd_flip,
    handle_backend,
    center_kernel,
    check_nonnegativity_eigenvalues,
    to_torch,
)
from torchdr.affinity import GibbsAffinity


class PCA(DRModule):
    def __init__(self, n_components=2, device=None, verbose=False):
        super().__init__(n_components=n_components, device=device, verbose=verbose)

    def fit(self, X: torch.Tensor):
        X = to_torch(X, device=self.device, verbose=self.verbose)
        self.mean_ = X.mean(0, keepdim=True)
        U, _, V = torch.linalg.svd(X - self.mean_, full_matrices=False)
        U, V = svd_flip(U, V)  # flip eigenvectors' sign to enforce deterministic output
        self.components_ = V[: self.n_components]
        return self

    @handle_backend
    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


# inspired from sklearn.decomposition.KernelPCA
class KernelPCA(DRModule):
    def __init__(
        self,
        affinity=GibbsAffinity(),
        n_components=2,
        verbose=False,
        remove_zero_eig=False,
    ):
        super().__init__(n_components=n_components, verbose=verbose)
        self.affinity = affinity
        self.remove_zero_eig = remove_zero_eig

    def fit(self, X):
        super().fit(X)
        K = self.affinity.fit_transform(X)
        K = center_kernel(K)
        K = aslinearoperator(K)

        # compute eigendecomposition
        self.eigenvalues_, self.eigenvectors_ = eigsh(K, k=self.n_components)

        # make sure that the eigenvalues are ok and fix numerical issues
        self.eigenvalues_ = check_nonnegativity_eigenvalues(self.eigenvalues_)

        # flip eigenvectors' sign to enforce deterministic output
        self.eigenvectors_, _ = svd_flip(
            self.eigenvectors_, torch.zeros_like(self.eigenvectors_).T
        )

        # sort eigenvectors in descending order
        indices = self.eigenvalues_.argsort(descending=True)
        self.eigenvalues_ = self.eigenvalues_[indices]
        self.eigenvectors_ = self.eigenvectors_[:, indices]

        # remove eigenvectors with a zero eigenvalue (null space) if required
        if self.remove_zero_eig or self.n_components is None:
            self.eigenvectors_ = self.eigenvectors_[:, self.eigenvalues_ > 0]
            self.eigenvalues_ = self.eigenvalues_[self.eigenvalues_ > 0]

        self.eigenvectors_ = self.eigenvectors_[:, -self.n_components :]
        return self

    @handle_backend
    def transform(self, X):
        K = self.affinity.fit_transform(X)
        K = center_kernel(K)
        return (
            K
            @ self.eigenvectors_
            @ torch.diag(1 / self.eigenvalues_[-self.n_components :]).sqrt()
        )

    @handle_backend
    def fit_transform(self, X):
        self.fit(X)
        print(self.eigenvectors_.shape, self.eigenvalues_.shape)
        print(self.eigenvalues_)
        return self.eigenvectors_ * self.eigenvalues_[-self.n_components :].sqrt()
