"""Tools for optimization problems."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause License

import torch
from torchdr.utils import compile_if_requested

DTYPE = torch.float32
DEVICE = "cpu"


@compile_if_requested
def binary_search(
    f,
    n,
    begin=None,
    end=None,
    max_iter=1000,
    tol=1e-9,
    verbose=False,
    dtype=DTYPE,
    device=DEVICE,
    logger=None,
    compile: bool = False,
):
    r"""Implement the binary search root finding method.

    Perform a batched binary search to find the root of an increasing function f.
    The domain of f is restricted to positive floats.

    Parameters
    ----------
    f : function :math:`\mathbb{R}_{>0} \to \mathbb{R}`
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
    dtype : torch.dtype, optional
        data type of the input.
    device : str, optional
        device on which the computation is performed.
    logger : logging.Logger, optional
        logger to use for printing.
    compile : bool, optional
        if True, the function is compiled.

    Returns
    -------
    m : torch.Tensor of shape (n)
        root of f.
    """
    begin, end = init_bounds(
        f=f,
        n=n,
        begin=begin,
        end=end,
        dtype=dtype,
        device=device,
        verbose=verbose,
        logger=logger,
    )

    m = (begin + end) / 2
    fm = f(m)

    i = 0
    for i in range(max_iter):
        if torch.max(torch.abs(fm)) < tol:
            break

        sam = fm * f(begin) > 0
        begin = sam * m + (~sam) * begin
        end = (~sam) * m + sam * end
        m = (begin + end) / 2
        fm = f(m)

    if verbose:
        n_iter = i + 1 if max_iter > 0 else 0
        msg = f"Root found in {n_iter} iterations."
        if logger is None:
            print("[TorchDR] " + msg)
        else:
            logger.info(msg)

    return m


@compile_if_requested
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
    logger=None,
    compile: bool = False,
):
    r"""Implement the false position root finding method.

    Perform a batched false position method to find the root
    of an increasing function f.
    The domain of f is restricted to positive floats.

    Parameters
    ----------
    f : function :math:`\mathbb{R}_{>0} \to \mathbb{R}`
        increasing function which root should be computed.
    n : int
        size of the input of f.
    begin : torch.Tensor of shape (n) or float, optional
        initial lower bound of the root.
    end : torch.Tensor of shape (n) or float, optional
        initial upper bound of the root.
    max_iter : int, optional
        maximum iterations of search.
    tol : float, optional
        precision threshold at which the algorithm stops.
    verbose : bool, optional
        if True, prints current bounds.
    dtype : torch.dtype, optional
        data type of the input.
    device : str, optional
        device on which the computation is performed.
    logger : logging.Logger, optional
        logger to use for printing.
    compile : bool, optional
        if True, the function is compiled.

    Returns
    -------
    m : torch.Tensor of shape (n)
        root of f.
    """
    begin, end = init_bounds(
        f=f,
        n=n,
        begin=begin,
        end=end,
        dtype=dtype,
        device=device,
        verbose=verbose,
        logger=logger,
    )

    f_begin, f_end = f(begin), f(end)
    m = begin - ((begin - end) / (f(begin) - f(end))) * f(begin)
    fm = f(m)
    assert m.shape == begin.shape == end.shape, (
        "dimension changed after evaluating the function which root should be computed."
    )

    i = 0
    for i in range(max_iter):
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
        n_iter = i + 1 if max_iter > 0 else 0
        msg = f"Root found in {n_iter} iterations."
        if logger is None:
            print("[TorchDR] " + msg)
        else:
            logger.info(msg)

    return m


@compile_if_requested
def init_bounds(
    f,
    n,
    begin=None,
    end=None,
    dtype=DTYPE,
    device=DEVICE,
    verbose=True,
    logger=None,
    compile: bool = False,
):
    """Initialize the bounds of the root search."""
    if begin is None:
        begin = torch.ones(n, dtype=dtype, device=device)
    else:
        assert isinstance(begin, (int, float, torch.Tensor)), (
            "begin must be a float, an int or a tensor."
        )
        if isinstance(begin, torch.Tensor):
            assert begin.shape == (n,), "begin must have the same shape as the output."
            begin = begin.to(dtype=dtype, device=device)
        begin = begin * torch.ones(n, dtype=dtype, device=device)

    if end is None:
        end = torch.ones(n, dtype=dtype, device=device)
    else:
        assert isinstance(end, (int, float, torch.Tensor)), (
            "end must be a float, an int or a tensor."
        )
        if isinstance(end, torch.Tensor):
            assert end.shape == (n,), "end must have the same shape as the output."
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

    if verbose and eval_counter > 0:
        msg = f"Root search bounds initialized in {eval_counter} evaluation(s)."
        if logger is None:
            print("[TorchDR] " + msg)
        else:
            logger.info(msg)

    return begin, end
