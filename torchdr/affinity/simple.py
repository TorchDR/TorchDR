# -*- coding: utf-8 -*-
"""
Common simple affinities
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
import pykeops
import numpy as np

from torchdr.utils import to_torch
from torchdr.affinity.base import (
    Affinity,
    TransformableLogAffinity,
)
from torchdr.utils import (
    inputs_to_torch,
)


class GaussianAffinity(TransformableLogAffinity):
    r"""
    Computes the Gaussian affinity matrix :math:`\exp( - \mathbf{C} / \sigma)`
    where :math:`\mathbf{C}` is the pairwise distance matrix and
    :math:`\sigma` is the bandwidth parameter.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth parameter.
    metric : str, optional
        Metric to use for pairwise distances computation.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
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
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        self.sigma = sigma

    def _log_affinity_formula(self, C: torch.Tensor | pykeops.torch.LazyTensor):
        return -C / self.sigma


class StudentAffinity(TransformableLogAffinity):
    r"""
    Computes the Student affinity matrix based on the Student-t distribution:

    .. math::
        \left(1 + \frac{\mathbf{C}}{\nu}\right)^{-\frac{\nu + 1}{2}}

    where :math:`\nu > 0` is the degrees of freedom parameter.

    Parameters
    ----------
    degrees_of_freedom : int, optional
        Degrees of freedom for the Student-t distribution.
    metric : str, optional
        Metric to use for pairwise distances computation.
    zero_diag : bool, optional
        Whether to set the diagonal of the affinity matrix to zero.
    device : str, optional
        Device to use for computations.
    keops : bool, optional
        Whether to use KeOps for computations.
    verbose : bool, optional
        Verbosity.
    """

    def __init__(
        self,
        degrees_of_freedom: int = 1,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        self.degrees_of_freedom = degrees_of_freedom

    def _log_affinity_formula(self, C: torch.Tensor | pykeops.torch.LazyTensor):
        return (
            -0.5
            * (self.degrees_of_freedom + 1)
            * (C / self.degrees_of_freedom + 1).log()
        )


class ScalarProductAffinity(Affinity):
    r"""
    Computes the scalar product affinity matrix :math:`\mathbf{X} \mathbf{X}^\top`
    where :math:`\mathbf{X}` is the input data. The affinity can be normalized
    according to the specified normalization dimensions.

    Parameters
    ----------
    normalization_dim : int or Tuple[int], optional
        Dimension along which to normalize the affinity matrix. Default is None.
    device : str, optional
        Device to use for computations. Default is "cuda".
    keops : bool, optional
        Whether to use KeOps for computations. Default is True.
    verbose : bool, optional
        Verbosity. Default is True.
    centering : bool, optional
        Whether to center the data by subtracting the mean. Default is False.
    """

    def __init__(
        self,
        device: str = "cuda",
        keops: bool = False,
        verbose: bool = True,
        centering: bool = False,
    ):
        super().__init__(
            metric="angular",
            device=device,
            keops=keops,
            verbose=verbose,
            zero_diag=False,
        )
        self.centering = centering

    def fit(self, X: torch.Tensor | np.ndarray):
        r"""
        Fits the scalar product affinity model to the provided data.

        This method computes the scalar product affinity matrix
        :math:`\mathbf{X} \mathbf{X}^T` for the input data. If centering is
        enabled, the data is centered by subtracting the mean before computing
        the affinity matrix.
        The affinity matrix is then normalized according to the specified
        normalization dimensions.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : ScalarProductAffinity
            The fitted scalar product affinity model.
        """
        self.data_ = to_torch(X, device=self.device, verbose=self.verbose)
        if self.centering:
            self.data_ = self.data_ - self.data_.mean(0)
        self.affinity_matrix_ = -self._distance_matrix(self.data_)

        return self

    @inputs_to_torch
    def transform(
        self,
        X: torch.Tensor | np.ndarray,
        Y: torch.Tensor | np.ndarray = None,
        indices: torch.Tensor = None,
    ):
        r"""
        Computes the scalar product affinity between X and Y.
        If Y is None, sets Y = X.
        If indices is not None, the output has shape (n, k) and its (i,j) element is the
        affinity between X[i] and Y[indices[i, j]].

        Parameters
        ----------
        X : torch.Tensor or np.ndarray of shape (n_samples_x, n_features)
            Input data.
        Y : torch.Tensor or np.ndarray of shape (n_samples_y, n_features), optional
            Second Input data. Default is None.
        indices : torch.Tensor of shape (n_samples_x, batch_size), optional
            Indices of pairs to compute. Default is None.

        Returns
        -------
        P : torch.Tensor or pykeops.torch.LazyTensor
            Scalar product between X and Y.
        """
        C = self._distance_matrix_transform(X, Y=Y, indices=indices)
        return -C
