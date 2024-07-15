# -*- coding: utf-8 -*-
"""
Spectral methods for dimensionality reduction
"""

# Authors: Hugues Van Assel <vanasselhugues@gmail.com>
#          Mathurin Massias
#
# License: BSD 3-Clause License

import torch

from torchdr.base import DRModule
from torchdr.utils import (
    sum_red,
    svd_flip,
    handle_backend,
    center_kernel,
    check_nonnegativity_eigenvalues,
)
from torchdr.affinity import (
    Affinity,
    GaussianAffinity,
    UnnormalizedAffinity,
    UnnormalizedLogAffinity,
)


class PCA(DRModule):
    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        verbose: bool = True,
        random_state: float = 0,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )

    def fit(self, X: torch.Tensor):
        X = super().fit(X)
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
        affinity: Affinity = GaussianAffinity(),
        n_components: int = 2,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
        random_state: float = 0,
        nodiag: bool = False,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            keops=keops,
            verbose=verbose,
            random_state=random_state,
        )

        self.affinity = affinity
        self.affinity.keops = keops
        self.affinity.device = device
        self.affinity.random_state = random_state
        self.nodiag = nodiag

        if keops:
            raise NotImplementedError(
                "[TorchDR] ERROR : KeOps is not (yet) supported for KernelPCA."
            )

    def fit(self, X):
        X = super().fit(X)
        self.X_fit_ = X.clone()
        K = self.affinity(X)
        K, _, col_mean, mean = center_kernel(K, return_all=True)
        self.K_fit_rows_ = col_mean
        self.K_fit_all_ = mean

        # compute eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(K)

        # make sure that the eigenvalues are ok and fix numerical issues
        eigvals = check_nonnegativity_eigenvalues(eigvals)

        # flip eigenvectors' sign to enforce deterministic output
        eigvecs, _ = svd_flip(eigvecs, torch.zeros_like(eigvecs).T)

        # sort eigenvectors in descending order (torch eigvals are increasing)
        eigvals = torch.flip(eigvals, dims=(0,))
        eigvecs = torch.flip(eigvecs, dims=(1,))

        # remove eigenvectors with a zero eigenvalue (null space) if required
        if self.nodiag or self.n_components is None:
            eigvecs = eigvecs[:, eigvals > 0]
            eigvals = eigvals[eigvals > 0]

        eigvecs = eigvecs[:, : self.n_components]

        self.eigenvectors_ = eigvecs
        self.eigenvalues_ = eigvals
        return self

    @handle_backend
    def transform(self, X):
        if not isinstance(
            self.affinity, (UnnormalizedAffinity, UnnormalizedLogAffinity)
        ):
            aff_name = self.affinity.__class__.__name__
            raise ValueError(
                "KernelPCA.transform can only be used when `affinity` is "
                "an UnnormalizedAffinity or UnnormalizedLogAffinity. "
                f"{aff_name} is not. Use the fit_transform method instead."
            )
        K = self.affinity(X, self.X_fit_)
        # K = center_kernel(K)
        # center à la sklearn: using fit data for rows and all, new data for col
        pred_cols = sum_red(K, 1) / self.K_fit_rows_.shape[1]
        K -= self.K_fit_rows_
        K -= pred_cols
        K += self.K_fit_all_

        result = (
            K
            @ self.eigenvectors_
            @ torch.diag(1 / self.eigenvalues_[: self.n_components]).sqrt()
        )
        # remove np.inf arising from division by 0 eigenvalues:
        zero_eigvals = self.eigenvalues_[: self.n_components] == 0
        if zero_eigvals.any():
            result[:, zero_eigvals] = 0
        return result

    @handle_backend
    def fit_transform(self, X):
        self.fit(X)
        return self.eigenvectors_ * self.eigenvalues_[: self.n_components].sqrt()
