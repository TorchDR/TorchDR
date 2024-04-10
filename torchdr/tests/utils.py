# -*- coding: utf-8 -*-
"""
Useful functions for testing
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from torch.testing import assert_close

from pykeops.torch import LazyTensor


def check_equality_torch_keops(P, P_keops, K=None, tol=1e-5):
    """
    Check that a torch.Tensor and a LazyTensor are equal on their largest entries.
    """
    assert isinstance(P, torch.Tensor), "P is not a torch.Tensor"
    assert isinstance(P_keops, LazyTensor), "P_keops is not a LazyTensor"
    assert P.shape == P_keops.shape, "P and P_keops do not have the same shape"

    n = P.shape[0]
    if K is None:
        K = n
    else:
        assert K <= n, "K is larger than the number of rows of P"

    # retrieve largest P_keops values and arguments
    Largest_keops_arg = (-P_keops).argKmin(K=K, dim=1)
    Largest_keops_values = -(-P_keops).Kmin(K=K, dim=1)

    # retrieve largest P values and arguments
    topk = P.topk(K, dim=1)
    Largest_arg = topk.indices
    Largest_values = topk.values

    # check that the largest values are the same
    assert_close(
        Largest_values,
        Largest_keops_values,
        atol=tol,
        rtol=tol,
        msg="Torch and Keops largest values are different",
    )
    assert_close(
        Largest_arg,
        Largest_keops_arg,
        atol=tol,
        rtol=tol,
        msg="Torch and Keops largest arguments are different",
    )
