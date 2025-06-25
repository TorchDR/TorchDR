"""Kernel Principal Component Analysis module."""

# Authors: Hugues Van Assel <vanasselhugues@gmail.com>
#          Mathurin Massias
#
# License: BSD 3-Clause License

from typing import Union, Any, Optional

import numpy as np
import torch

from torchdr.base import DRModule
from torchdr.utils import (
    handle_type,
    to_torch,
    svd_flip,
    sum_red,
    center_kernel,
    check_nonnegativity_eigenvalues,
    log_with_timing,
)

from torchdr.affinity import (
    Affinity,
    GaussianAffinity,
    UnnormalizedAffinity,
    UnnormalizedLogAffinity,
)


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
    backend : {"keops", "faiss", None}, optional
        Which backend to use for handling sparsity and memory efficiency.
        Default is None.
    verbose : bool, default=False
        Whether to print information during the computations.
    random_state : float, default=None
        Random seed for reproducibility.
    nodiag : bool, default=False
        Whether to remove eigenvectors with a zero eigenvalue.
    """

    def __init__(
        self,
        affinity: Affinity = GaussianAffinity(),
        n_components: int = 2,
        device: str = "auto",
        backend: str = None,
        verbose: bool = False,
        random_state: float = None,
        nodiag: bool = False,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
            **kwargs,
        )

        self.affinity = affinity
        self.affinity.backend = backend
        self.affinity.device = device
        self.nodiag = nodiag

        if backend == "keops":
            raise NotImplementedError(
                "[TorchDR] ERROR : KeOps is not (yet) supported for KernelPCA."
            )

    @log_with_timing(log_device_backend=True)
    def fit(self, X: Union[torch.Tensor, np.ndarray]):
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
        self.embedding_ = (
            self.eigenvectors_ * self.eigenvalues_[: self.n_components].sqrt()
        )
        return self

    @handle_type
    def transform(self, X: Union[torch.Tensor, np.ndarray]):
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

    def _fit_transform(self, X: torch.Tensor, y: Optional[Any] = None):
        r"""Fit the KernelPCA model and project the input data onto the components.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Data on which to fit the KernelPCA model and project onto the components.
        y : Optional[Any], default=None
            Ignored in this method.

        Returns
        -------
        X_new : torch.Tensor or np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
        self.fit(X)
        return self.embedding_
