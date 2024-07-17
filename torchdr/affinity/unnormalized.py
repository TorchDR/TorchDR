# -*- coding: utf-8 -*-
"""Common simple affinities."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

from torchdr.utils import LazyTensorType
from torchdr.affinity.base import UnnormalizedAffinity, UnnormalizedLogAffinity


class GaussianAffinity(UnnormalizedLogAffinity):
    r"""Compute the Gaussian affinity matrix.

    Its expression is as follows : :math:`\exp( - \mathbf{C} / \sigma)`
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

    def _log_affinity_formula(self, C: torch.Tensor | LazyTensorType):
        return -C / self.sigma


class StudentAffinity(UnnormalizedLogAffinity):
    r"""Compute the Student affinity matrix based on the Student-t distribution.

    Its expression is given by:

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
        Verbosity. Default is False.
    """

    def __init__(
        self,
        degrees_of_freedom: int = 1,
        metric: str = "sqeuclidean",
        zero_diag: bool = True,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            metric=metric,
            zero_diag=zero_diag,
            device=device,
            keops=keops,
            verbose=verbose,
        )
        self.degrees_of_freedom = degrees_of_freedom

    def _log_affinity_formula(self, C: torch.Tensor | LazyTensorType):
        return (
            -0.5
            * (self.degrees_of_freedom + 1)
            * (C / self.degrees_of_freedom + 1).log()
        )


class ScalarProductAffinity(UnnormalizedAffinity):
    r"""Compute the scalar product affinity matrix.

    Its expression is given by :math:`\mathbf{X} \mathbf{X}^\top`
    where :math:`\mathbf{X}` is the input data.

    Parameters
    ----------
    device : str, optional
        Device to use for computations. Default is "cuda".
    keops : bool, optional
        Whether to use KeOps for computations. Default is True.
    verbose : bool, optional
        Verbosity. Default is False.
    """

    def __init__(
        self,
        device: str = "auto",
        keops: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            metric="angular",
            device=device,
            keops=keops,
            verbose=verbose,
            zero_diag=False,
        )

    def _affinity_formula(self, C: torch.Tensor | LazyTensorType):
        return -C
