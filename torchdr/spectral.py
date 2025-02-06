"""Spectral methods for dimensionality reduction."""

# Authors: Hugues Van Assel <vanasselhugues@gmail.com>
#          Mathurin Massias
#          @sirluk
#
# License: BSD 3-Clause License

from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch

from torchdr.affinity import (
    Affinity,
    GaussianAffinity,
    UnnormalizedAffinity,
    UnnormalizedLogAffinity,
)
from torchdr.base import DRModule
from torchdr.utils import (
    center_kernel,
    check_nonnegativity_eigenvalues,
    handle_type,
    sum_red,
    svd_flip,
    to_torch,
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
        U, S, V = torch.linalg.svd(
            X - self.mean_, full_matrices=False, driver=self.svd_driver
        )
        U, V = svd_flip(U, V)  # flip eigenvectors' sign to enforce deterministic output
        self.components_ = V[: self.n_components]
        self.embedding_ = U[:, : self.n_components] @ S[: self.n_components].diag()
        return self

    @handle_type
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
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            backend=backend,
            verbose=verbose,
            random_state=random_state,
        )

        self.affinity = affinity
        self.affinity.backend = backend
        self.affinity.device = device
        self.affinity.random_state = random_state
        self.nodiag = nodiag

        if backend == "keops":
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

    @handle_type
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


class IncrementalPCA(DRModule):
    """Incremental Principal Components Analysis (IPCA) leveraging PyTorch for GPU acceleration.

    This class provides methods to fit the model on data incrementally in batches,
    and to transform new data based on the principal components learned during the fitting process.

    It is partially useful when the dataset to be decomposed is too large to fit in memory.
    Adapted from `Scikit-learn Incremental PCA <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py>`_.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If `None`, it's set to the minimum of the
        number of samples and features. Defaults to None.
    copy : bool
        If False, input data will be overwritten. Defaults to True.
    batch_size : int, optional
        The number of samples to use for each batch. Only needed if self.fit is called.
        If `None`, it's inferred from the data and set to `5 * n_features`.
        Defaults to None.
    svd_driver : str, optional
        Name of the cuSOLVER method to be used for torch.linalg.svd.
        This keyword argument only works on CUDA inputs.
        Available options are: None, gesvd, gesvdj and gesvda.
        Defaults to None.
    lowrank : bool, optional
        Whether to use torch.svd_lowrank instead of torch.linalg.svd which can be faster.
        Defaults to False.
    lowrank_q : int, optional
        For an adequate approximation of n_components, this parameter defaults to
        n_components * 2.
    lowrank_niter : int, optional
        Number of subspace iterations to conduct for torch.svd_lowrank.
        Defaults to 4.
    device : str, optional
        Device on which the computations are performed. Defaults to "auto".
    """  # noqa: E501

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: Optional[bool] = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4,
        device: str = "auto",
        verbose: bool = False,
        random_state: float = None,
    ):
        super().__init__(
            n_components=n_components,
            device=device,
            verbose=verbose,
            random_state=random_state,
        )
        self.copy = copy
        self.batch_size = batch_size
        self.n_features_ = None

        if lowrank:
            if lowrank_q is None:
                assert n_components is not None, (
                    "n_components must be specified when using lowrank mode "
                )
                "with lowrank_q=None."
                lowrank_q = n_components * 2
            assert lowrank_q >= n_components, (
                "lowrank_q must be greater than or equal to n_components."
            )

            def svd_fn(X):
                U, S, V = torch.svd_lowrank(X, q=lowrank_q, niter=lowrank_niter)
                return U, S, V.mH  # V is returned as a conjugate transpose

            self._svd_fn = svd_fn

        else:
            self._svd_fn = partial(
                torch.linalg.svd, full_matrices=False, driver=svd_driver
            )

    def _validate_data(self, X) -> torch.Tensor:
        valid_dtypes = [torch.float32, torch.float64]

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        elif self.copy:
            X = X.clone()

        n_samples, n_features = X.shape
        if self.n_components is None:
            pass
        elif self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} invalid for "
                "n_features={n_features}, need more rows than columns "
                "for IncrementalPCA processing."
            )
        elif self.n_components > n_samples:
            raise ValueError(
                f"n_components={self.n_components} must be less or equal "
                "to the batch number of samples {n_samples}."
            )

        if X.dtype not in valid_dtypes:
            X = X.to(torch.float32)

        return X

    @staticmethod
    def _incremental_mean_and_var(
        X, last_mean, last_variance, last_sample_count
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        if last_sample_count > 0:
            assert last_mean is not None, (
                "last_mean should not be None if last_sample_count > 0."
            )
            assert last_variance is not None, (
                "last_variance should not be None if last_sample_count > 0."
            )

        new_sample_count = torch.tensor([X.shape[0]], device=X.device)
        updated_sample_count = last_sample_count + new_sample_count

        if last_mean is None:
            last_sum = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
        else:
            last_sum = last_mean * last_sample_count

        new_sum = X.sum(dim=0, dtype=torch.float64)

        updated_mean = (last_sum + new_sum) / updated_sample_count

        T = new_sum / new_sample_count
        temp = X - T
        correction = temp.sum(dim=0, dtype=torch.float64).square()
        temp.square_()
        new_unnormalized_variance = temp.sum(dim=0, dtype=torch.float64)
        new_unnormalized_variance -= correction / new_sample_count
        if last_variance is None:
            updated_variance = new_unnormalized_variance / updated_sample_count
        else:
            last_unnormalized_variance = last_variance * last_sample_count
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum).square()
            )
            updated_variance = updated_unnormalized_variance / updated_sample_count

        return updated_mean, updated_variance, updated_sample_count

    def fit(self, X: torch.Tensor | np.ndarray, check_input: bool = True):
        """Fit the model with data `X` using minibatches of size `batch_size`.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            The input data tensor with shape (n_samples, n_features).
        check_input : bool, optional
            If True, validates the input. Defaults to True.

        Returns
        -------
        IncrementalPCA:
            The fitted IPCA model.
        """
        X = to_torch(X, device="auto")
        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size = 5 * n_features

        for batch in self.gen_batches(
            n_samples, self.batch_size, min_batch_size=self.n_components or 0
        ):
            X_batch = X[batch].to(X.device if self.device == "auto" else self.device)
            self.partial_fit(X_batch, check_input=False)

        return self

    def partial_fit(self, X, check_input=True):
        """Fit incrementally the model with batch data `X`.

        Parameters
        ----------
        X : torch.Tensor
            The batch input data tensor with shape (n_samples, n_features).
        check_input : bool, optional
            If True, validates the input. Defaults to True.

        Returns
        -------
        IncrementalPCA:
            The updated IPCA model after processing the batch.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        # Initialize attributes to avoid errors during the first call to partial_fit
        if first_pass:
            self.mean_ = None
            self.var_ = None
            self.n_samples_seen_ = torch.tensor([0], device=X.device)
            self.n_features_ = n_features
            if not self.n_components:
                self.n_components = min(n_samples, n_features)

        if n_features != self.n_features_:
            raise ValueError(
                "Number of features of the new batch does not match "
                "the number of features of the first batch."
            )

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, self.n_samples_seen_
        )

        if first_pass:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                (self.n_samples_seen_ / n_total_samples) * n_samples
            )
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)
            X = torch.vstack(
                (
                    self.singular_values_.view((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        U, S, Vt = self._svd_fn(X)
        U, Vt = svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[: self.n_components]
        self.singular_values_ = S[: self.n_components]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]
        if self.n_components not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components :].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0, device=X.device)
        return self

    @handle_type
    def transform(self, X: torch.Tensor | np.ndarray):
        """Apply dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components
        previously extracted from a training set.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            New data tensor with shape (n_samples, n_features) to be transformed.

        Returns
        -------
        torch.Tensor:
            Transformed data tensor with shape (n_samples, n_components).
        """
        X = X - self.mean_
        return X @ self.components_.T

    def fit_transform(self, X: torch.Tensor | np.ndarray):
        r"""Fit the model with data `X` and apply the dimensionality reduction.

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
        return self.transform(X)

    @staticmethod
    def gen_batches(n: int, batch_size: int, min_batch_size: int = 0):
        """Generate slices containing `batch_size` elements from 0 to `n`.

        The last slice may contain less than `batch_size` elements, when
        `batch_size` does not divide `n`.

        Parameters
        ----------
        n : int
            Size of the sequence.
        batch_size : int
            Number of elements in each batch.
        min_batch_size : int, optional
            Minimum number of elements in each batch. Defaults to 0.

        Yields
        ------
        slice:
            A slice of `batch_size` elements.
        """
        start = 0
        for _ in range(int(n // batch_size)):
            end = start + batch_size
            if end + min_batch_size > n:
                continue
            yield slice(start, end)
            start = end
        if start < n:
            yield slice(start, n)
