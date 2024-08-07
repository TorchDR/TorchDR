# -*- coding: utf-8 -*-
"""Losses to define DR objectives."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

from torchdr.utils.wrappers import sum_all_axis_except_batch


@sum_all_axis_except_batch
def cross_entropy_loss(P, Q, log=False):
    r"""Compute the cross-entropy between P and Q.

    Supports log domain input for Q.
    """
    if log:
        return -P * Q
    else:
        return -P * Q.log()


@sum_all_axis_except_batch
def square_loss(P, Q):
    r"""Compute the square loss between P and Q."""
    return (P - Q) ** 2
