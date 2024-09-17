# -*- coding: utf-8 -*-
"""Spectral methods for dimensionality reduction."""

# Authors: Hugues Van Assel <vanasselhugues@gmail.com>
#          Mathurin Massias
#
# License: BSD 3-Clause License

import torch
import numpy as np

from torchdr.base import DRModule
from torchdr.utils import (
    to_torch,
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
    r"""Principal Component Analysis module.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to project the input data onto.
    device : str, default="auto"
        Device on which the computations are performed.
    verbose : bool, default=False
        Whether to print information during the computations.
    random_state : float, default=0
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        verbose: bool = False,
        random_state: float = 0,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""Fit the PCA model.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which to fit the PCA model.

        Returns
        -------
        self : PCA
            The fitted PCA model.
        """
        X = to_torch(X, device=self.device)
        self.mean_ = X.mean(0, keepdim=True)
        U, S, V = torch.linalg.svd(X - self.mean_, full_matrices=False)
        U, V = svd_flip(U, V)  # flip eigenvectors' sign to enforce deterministic output
        self.components_ = V[: self.n_components]
        self.embedding_ = U[:, : self.n_components] @ S[: self.n_components].diag()
        return self

    @handle_backend
    def transform(self, X: torch.Tensor | np.ndarray):
        r"""Project the input data onto the PCA components.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data to project onto the PCA components.

        Returns
        -------
        X_new : torch.Tensor or np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: torch.Tensor | np.ndarray):
        r"""Fit the PCA model and project the input data onto the components.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which to fit the PCA model and project onto the components.

        Returns
        -------
        X_new : torch.Tensor or np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
        self.fit(X)
        return self.embedding_


class KernelPCA(DRModule):
    r"""Kernel Principal Component Analysis module.

    Parameters
    ----------
    affinity : Affinity, default=GaussianAffinity()
        Affinity object to compute the kernel matrix.
    n_components : int, default=2
        Number of components to project the input data onto.
    device : str, default="auto"
        Device on which the computations are performed.
    keops : bool, default=False
        Whether to use KeOps for computations.
    verbose : bool, default=False
        Whether to print information during the computations.
    random_state : float, default=0
        Random seed for reproducibility.
    nodiag : bool, default=False
        Whether to remove eigenvectors with a zero eigenvalue.
    """

    def __init__(
        self,
        affinity: Affinity = GaussianAffinity(),
        n_components: int = 2,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
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

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""Fit the KernelPCA model.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which to fit the KernelPCA model.

        Returns
        -------
        self : KernelPCA
            The fitted KernelPCA model.
        """
        X = to_torch(X, device=self.device)
        self.X_fit_ = X.clone()
        K = self.affinity(X)
        K, _, col_mean, mean = center_kernel(K, return_all=True)
        self.K_fit_rows_ = col_mean
        self.K_fit_all_ = mean

        eigvals, eigvecs = torch.linalg.eigh(K)
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
    def transform(self, X: torch.Tensor | np.ndarray):
        r"""Project the input data onto the KernelPCA components.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data to project onto the KernelPCA components.

        Returns
        -------
        X_new : torch.Tensor or np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
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

    def fit_transform(self, X: torch.Tensor | np.ndarray):
        r"""Fit the KernelPCA model and project the input data onto the components.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which to fit the KernelPCA model and project onto the components.

        Returns
        -------
        X_new : torch.Tensor or np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
        self.fit(X)
        return self.eigenvectors_ * self.eigenvalues_[: self.n_components].sqrt()
