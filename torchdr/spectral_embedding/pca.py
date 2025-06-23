"""Principal Component Analysis module."""

# Authors: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from typing import Optional, Union

import numpy as np
import torch

from torchdr.base import DRModule
from torchdr.utils import handle_type, seed_everything, to_torch, svd_flip


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
    random_state : float, default=None
        Random seed for reproducibility.
    svd_driver : str, optional
        Name of the cuSOLVER method to be used for torch.linalg.svd.
        This keyword argument only works on CUDA inputs.
        Available options are: None, gesvd, gesvdj and gesvda.
        Defaults to None.
    """

    def __init__(
        self,
        n_components: int = 2,
        device: str = "auto",
        verbose: bool = False,
        random_state: float = None,
        svd_driver: Optional[str] = None,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )
        self.svd_driver = svd_driver

    def fit(self, X: Union[torch.Tensor, np.ndarray]):
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
        U, S, V = torch.linalg.svd(
            X - self.mean_, full_matrices=False, driver=self.svd_driver
        )
        U, V = svd_flip(U, V)  # flip eigenvectors' sign to enforce deterministic output
        self.components_ = V[: self.n_components]
        self.embedding_ = U[:, : self.n_components] @ S[: self.n_components].diag()
        return self

    @handle_type
    def transform(self, X: Union[torch.Tensor, np.ndarray]):
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

    def fit_transform(self, X: Union[torch.Tensor, np.ndarray]):
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
        seed_everything(self.random_state)
        self.fit(X)
        return self.embedding_
