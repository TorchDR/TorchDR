import torch
from functools import partial

from typing import Optional, Tuple

from torchdr.utils import svd_flip


class IncrementalPCA:
    """Incremental Principal Components Analysis (IPCA) leveraging PyTorch for GPU acceleration.

    This class provides methods to fit the model on data incrementally in batches,
    and to transform new data based on the principal components learned during the fitting process.

    It is partially useful when the dataset to be decomposed is too large to fit in memory.
    Adapted from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py

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
        Name of the cuSOLVER method to be used for torch.linalg.svd. This keyword
        argument only works on CUDA inputs. Available options are: None, gesvd, gesvdj,
        and gesvda. Defaults to None.
    lowrank : bool, optional
        Whether to use torch.svd_lowrank instead of torch.linalg.svd which can be faster.
        Defaults to False.
    lowrank_q : int, optional
        For an adequate approximation of n_components, this parameter defaults to
        n_components * 2.
    lowrank_niter : int, optional
        Number of subspace iterations to conduct for torch.svd_lowrank.
        Defaults to 4.
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
    ):
        self.n_components_ = n_components
        self.copy = copy
        self.batch_size = batch_size
        self.n_features_ = None

        if lowrank:
            if lowrank_q is None:
                assert (
                    n_components is not None
                ), "n_components must be specified when using lowrank mode "
                "with lowrank_q=None."
                lowrank_q = n_components * 2
            assert (
                lowrank_q >= n_components
            ), "lowrank_q must be greater than or equal to n_components."

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
        if self.n_components_ is None:
            pass
        elif self.n_components_ > n_features:
            raise ValueError(
                f"n_components={self.n_components_} invalid for "
                "n_features={n_features}, need more rows than columns "
                "for IncrementalPCA processing."
            )
        elif self.n_components_ > n_samples:
            raise ValueError(
                f"n_components={self.n_components_} must be less or equal "
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
            assert (
                last_mean is not None
            ), "last_mean should not be None if last_sample_count > 0."
            assert (
                last_variance is not None
            ), "last_variance should not be None if last_sample_count > 0."

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
            last_over_new_count = last_sample_count.double() / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum).square()
            )
            updated_variance = updated_unnormalized_variance / updated_sample_count

        return updated_mean, updated_variance, updated_sample_count

    def fit(self, X, check_input=True):
        """Fit the model with data `X` using minibatches of size `batch_size`.

        Parameters:
        ----------
        X : torch.Tensor
            The input data tensor with shape (n_samples, n_features).
        check_input : bool, optional
            If True, validates the input. Defaults to True.

        Returns:
        -------
        IncrementalPCA:
            The fitted IPCA model.
        """
        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size = 5 * n_features

        for batch in self.gen_batches(
            n_samples, self.batch_size, min_batch_size=self.n_components_ or 0
        ):
            self.partial_fit(X[batch], check_input=False)

        return self

    def partial_fit(self, X, check_input=True):
        """Fit incrementally the model with batch data `X`.

        Parameters:
        ----------
        X : torch.Tensor
            The batch input data tensor with shape (n_samples, n_features).
        check_input : bool, optional
            If True, validates the input. Defaults to True.

        Returns:
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
            if not self.n_components_:
                self.n_components_ = min(n_samples, n_features)

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
                (self.n_samples_seen_.double() / n_total_samples) * n_samples
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
        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_ :].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0, device=X.device)
        return self

    def transform(self, X) -> torch.Tensor:
        """Apply dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components
        previously extracted from a training set.

        Parameters:
        ----------
        X : torch.Tensor
            New data tensor with shape (n_samples, n_features) to be transformed.

        Returns:
        -------
        torch.Tensor:
            Transformed data tensor with shape (n_samples, n_components).
        """
        X = X - self.mean_
        return torch.mm(X.double(), self.components_.T).to(X.dtype)

    @staticmethod
    def gen_batches(n: int, batch_size: int, min_batch_size: int = 0):
        """Generate slices containing `batch_size` elements from 0 to `n`.

        The last slice may contain less than `batch_size` elements, when
        `batch_size` does not divide `n`.

        Parameters:
        ----------
        n : int
            Size of the sequence.
        batch_size : int
            Number of elements in each batch.
        min_batch_size : int, optional
            Minimum number of elements in each batch. Defaults to 0.

        Yields:
        -------
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
