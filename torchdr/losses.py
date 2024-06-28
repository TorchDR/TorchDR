# -*- coding: utf-8 -*-
"""
Losses to define DR objectives
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.utils import sum_all_axis


@sum_all_axis
def cross_entropy_loss(P, Q, log_Q=False):
    r"""
    Computes the cross-entropy between P and Q.
    Supports log domain input for Q.
    """
    if log_Q:
        return -P * Q
    else:
        return -P * Q.log()


@sum_all_axis
def binary_cross_entropy_loss(P, Q, coeff_repulsion=1):
    r"""
    Computes the binary cross-entropy between P and Q.
    Supports log domain input for Q.
    """
    return -P * Q.log() - coeff_repulsion * (1 - P) * (1 - Q).log()
    # return (
    #     -P * Q.clamp(1e-4, 1).log()
    #     - coeff_repulsion * (1 - P) * (1 - Q).clamp(1e-4, 1).log()
    # )


@sum_all_axis
def square_loss(P, Q):
    r"""
    Computes the square loss between P and Q.
    """
    return (P - Q) ** 2
