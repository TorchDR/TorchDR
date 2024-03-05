# -*- coding: utf-8 -*-
"""
Operators
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch


def entropy(P: torch.Tensor,
            log: bool = False,
            dim: int = -1):
    r"""
    Returns the entropy of P along axis dim, supports log domain input.

    Parameters
    ----------
    P: tensor
        input data.
    log: bool
        if True, assumes that P is in log domain.
    dim: int
        axis on which entropy is computed.
    """
    if log:
        return -(torch.exp(P)*(P-1)).sum(dim)
    else:
        return -(P*(torch.log(P)-1)).sum(dim)


def kl_div(P: torch.Tensor, Q: torch.Tensor, log: bool = False):
    r"""
    Returns the Kullback-Leibler divergence between P and Q, supports log domain input
    for both matrices.

    Parameters
    ----------
    P: tensor
        input data.
    Q: tensor
        input data.
    log: bool
        if True, assumes that P and Q are in log domain.
    """
    if log:
        return (torch.exp(P) * (P - Q - 1)).sum()
    else:
        return (P * (torch.log(P/Q) - 1)).sum()
