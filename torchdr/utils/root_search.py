"""Root search algorithms for solving scalar equations."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause License

import torch


from typing import Callable, Tuple

_DEFAULT_TOL = torch.tensor(1e-6)


@torch.compiler.disable
def binary_search(
    f: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    begin: float = 1.0,
    end: float = 1.0,
    max_iter: int = 100,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Batched binary search root finding.

    Finds the roots of an increasing function f over positive inputs
    by repeatedly narrowing the bracket [begin, end].

    Parameters
    ----------
    f : Callable[[torch.Tensor], torch.Tensor]
        Batched 1-D increasing function.
    n : int
        Batch size (length of the input/output vectors).
    begin : float, optional
        Scalar initial lower bound (default: 1.0).
    end : float, optional
        Scalar initial upper bound (default: 1.0).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    dtype : torch.dtype, optional
        Data type of all tensors (default: torch.float32).
    device : torch.device, optional
        Device for all tensors (default: CPU).

    Returns
    -------
    m : torch.Tensor of shape (n,)
        Estimated roots where |f(m)| < tol.
    """
    tol = _DEFAULT_TOL.to(device).to(dtype)
    b, e = init_bounds(f, n, begin, end, max_iter=max_iter, dtype=dtype, device=device)

    f_b = f(b)
    m = (b + e) * 0.5
    f_m = f(m)

    for _ in range(max_iter):
        active = torch.abs(f_m) >= tol
        if not active.any():
            break

        same_sign = f_m * f_b > 0

        mask1 = active & same_sign
        b[mask1] = m[mask1]
        f_b[mask1] = f_m[mask1]

        mask2 = active & (~same_sign)
        e[mask2] = m[mask2]

        m = (b + e) * 0.5
        f_m = f(m)

    return m


@torch.compiler.disable
def false_position(
    f: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    begin: float = 1.0,
    end: float = 1.0,
    max_iter: int = 100,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Batched false-position root finding.

    Uses linear interpolation to bracket and converge on the root
    of an increasing function f.

    Parameters
    ----------
    f : Callable[[torch.Tensor], torch.Tensor]
        Batched 1-D increasing function.
    n : int
        Batch size (length of the input/output vectors).
    begin : float, optional
        Scalar initial lower bound (default: 1.0).
    end : float, optional
        Scalar initial upper bound (default: 1.0).
    max_iter : int, optional
        Maximum number of iterations (default: 1000).
    dtype : torch.dtype, optional
        Data type of all tensors (default: torch.float32).
    device : torch.device, optional
        Device for all tensors (default: CPU).

    Returns
    -------
    m : torch.Tensor of shape (n,)
        Estimated roots where |f(m)| < tol.
    """
    tol = _DEFAULT_TOL.to(device).to(dtype)
    b, e = init_bounds(f, n, begin, end, max_iter=max_iter, dtype=dtype, device=device)

    f_b = f(b)
    f_e = f(e)
    m = b - (b - e) / (f_b - f_e) * f_b
    f_m = f(m)

    for _ in range(max_iter):
        active = torch.abs(f_m) >= tol
        if not active.any():
            break

        same_sign = f_m * f_b > 0

        mask1 = active & same_sign
        b[mask1] = m[mask1]
        f_b[mask1] = f_m[mask1]

        mask2 = active & (~same_sign)
        e[mask2] = m[mask2]
        f_e[mask2] = f_m[mask2]

        m = b - (b - e) / (f_b - f_e) * f_b
        f_m = f(m)

    return m


@torch.compiler.disable
def init_bounds(
    f: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    begin=1.0,
    end=1.0,
    max_iter=100,
    dtype: torch.dtype = torch.float32,
    device="cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize root‐search bounds for f, supporting both scalar and tensor inputs."""

    if isinstance(begin, torch.Tensor):
        b = begin.to(dtype=dtype, device=device)
        if b.shape != (n,):
            raise ValueError(f"begin tensor must have shape ({n},), got {b.shape}")
    else:
        b = torch.full((n,), float(begin), dtype=dtype, device=device)

    if isinstance(end, torch.Tensor):
        e = end.to(dtype=dtype, device=device)
        if e.shape != (n,):
            raise ValueError(f"end tensor must have shape ({n},), got {e.shape}")
    else:
        e = torch.full((n,), float(end), dtype=dtype, device=device)

    # shrink `b` downward until f(b) ≤ 0, pulling `e` in with it
    for _ in range(max_iter):
        mask = f(b) > 0
        if not mask.any():
            break

        old_b = b
        e = torch.where(mask, torch.min(e, old_b), e)
        b = torch.where(mask, b * 0.5, b)

    # expand `e` upward until f(e) ≥ 0, pushing `b` out with it
    for _ in range(max_iter):
        mask = f(e) < 0
        if not mask.any():
            break

        old_e = e
        b = torch.where(mask, torch.max(b, old_e), b)
        e = torch.where(mask, e * 2.0, e)

    return b, e
