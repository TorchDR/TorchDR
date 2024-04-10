# -*- coding: utf-8 -*-
"""
Tools for optimization problems
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause License

import torch
from tqdm import tqdm

DTYPE = torch.double
DEVICE = "cpu"

OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "NAdam": torch.optim.NAdam,
}


def binary_search(
    f,
    n,
    begin=None,
    end=None,
    max_iter=1000,
    tol=1e-9,
    verbose=False,
    dtype=DTYPE,
):
    r"""
    Performs a batched binary search to find the root of an increasing function f.

    Parameters
    ----------
    f : function :math:`\mathbb{R}^n \to \mathbb{R}^n`
        batched 1d increasing function which root should be computed.
    n : int
        size of the input of f.
    begin : float or torch.Tensor of shape (n), optional
        initial lower bound of the root.
    end : float or torch.Tensor of shape (n), optional
        initial upper bound of the root.
    max_iter : int, optional
        maximum iterations of search.
    tol : float, optional
        precision threshold at which the algorithm stops.
    verbose : bool, optional
        if True, prints current bounds.

    Returns
    -------
    m : tensor of shape (n)
        root of f.
    """
    begin, end = init_bounds(f=f, n=n, begin=begin, end=end, dtype=dtype)

    m = (begin + end) / 2
    fm = f(m)

    pbar = tqdm(range(max_iter), disable=not verbose)
    for _ in pbar:

        if torch.max(torch.abs(fm)) < tol:
            break

        sam = fm * f(begin) > 0
        begin = sam * m + (~sam) * begin
        end = (~sam) * m + sam * end
        m = (begin + end) / 2
        fm = f(m)

        if verbose:
            mean_f = fm.mean().item()
            std_f = fm.std().item()
            pbar.set_description(
                f"f mean : {float(mean_f): .2e}, "
                f"f std : {float(std_f): .2e}, "
                f"begin mean : {float(begin.mean().item()): .2e}, "
                f"end mean : {float(end.mean().item()): .2e} "
            )

    return m


def false_position(
    f,
    n,
    begin=None,
    end=None,
    max_iter=1000,
    tol=1e-9,
    verbose=False,
    dtype=DTYPE,
    device=DEVICE,
):
    r"""
    Performs a batched false position method to find the root
    of an increasing function f.

    Parameters
    ----------
    f : function :math:`\mathbb{R}^n \to \mathbb{R}^n`
        increasing function which root should be computed.
    n : int
        size of the input of f.
    begin : tensor of shape (n) or float, optional
        initial lower bound of the root.
    end : tensor of shape (n) or float, optional
        initial upper bound of the root.
    max_iter : int, optional
        maximum iterations of search.
    tol : float, optional
        precision threshold at which the algorithm stops.
    verbose : bool, optional
        if True, prints current bounds.
    dtype : torch.dtype, optional
        data type of the input.

    Returns
    -------
    m : tensor of shape (n)
        root of f.
    """
    begin, end = init_bounds(
        f=f, n=n, begin=begin, end=end, dtype=dtype, device=device, verbose=verbose
    )

    f_begin, f_end = f(begin), f(end)
    m = begin - ((begin - end) / (f(begin) - f(end))) * f(begin)
    fm = f(m)
    assert m.shape == begin.shape == end.shape

    pbar = tqdm(range(max_iter), disable=not verbose)
    for _ in pbar:

        if torch.max(torch.abs(fm)) < tol:
            break

        sam = fm * f_begin > 0
        begin = sam * m + (~sam) * begin
        f_begin = sam * fm + (~sam) * f_begin
        end = (~sam) * m + sam * end
        f_end = (~sam) * fm + sam * f_end
        m = begin - ((begin - end) / (f_begin - f_end)) * f_begin
        fm = f(m)

        if verbose:
            mean_f = fm.mean().item()
            std_f = fm.std().item()
            pbar.set_description(
                f"f mean : {float(mean_f): .2e}, "
                f"f std : {float(std_f): .2e}, "
                f"begin mean : {float(begin.mean().item()): .2e}, "
                f"end mean : {float(end.mean().item()): .2e} "
            )

    return m


def init_bounds(f, n, begin=None, end=None, dtype=DTYPE, device=DEVICE, verbose=True):
    """Initializes the bounds of the root search."""

    if begin is None:
        begin = torch.ones(n, dtype=dtype, device=device)
    else:
        assert isinstance(
            begin, (int, float, torch.Tensor)
        ), "begin must be a float, an int or a tensor."
        if isinstance(begin, torch.Tensor):
            begin = begin.to(dtype=dtype, device=device)
        begin = begin * torch.ones(n, dtype=dtype, device=device)

    if end is None:
        end = torch.ones(n, dtype=dtype, device=device)
    else:
        assert isinstance(
            end, (int, float, torch.Tensor)
        ), "end must be a float, an int or a tensor."
        if isinstance(end, torch.Tensor):
            end = end.to(dtype=dtype, device=device)
        end = end * torch.ones(n, dtype=dtype, device=device)

    eval_counter = 0

    # Ensure that begin lower bounds the root
    out_begin = f(begin) > 0
    while out_begin.any():
        end[out_begin] = torch.min(end[out_begin], begin[out_begin])
        begin[out_begin] /= 2
        out_begin = f(begin) > 0
        eval_counter += 1

    # Ensure that end upper bounds the root
    out_end = f(end) < 0
    while out_end.any():
        begin[out_end] = torch.max(begin[out_end], end[out_end])
        end[out_end] *= 2
        out_end = f(end) < 0
        eval_counter += 1

    if eval_counter and verbose:
        print(f"[TorchDR] {eval_counter} evaluations to set bounds of the root search.")

    return begin, end
