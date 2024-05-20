# -*- coding: utf-8 -*-
"""
Useful wrappers for dealing with backends and devices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import functools
import torch
import numpy as np
from pykeops.torch import LazyTensor


def contiguous_output(func):
    """
    Convert all output torch tensors to contiguous.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if isinstance(output, tuple):
            output = (
                out.contiguous() if isinstance(out, torch.Tensor) else out
                for out in output
            )
        elif isinstance(output, torch.Tensor):
            output = output.contiguous()
        return output

    return wrapper


@contiguous_output
def to_torch(x, device="cuda", verbose=True, return_backend_device=False):
    use_gpu = (device in ["cuda", "cuda:0", "gpu", None]) and torch.cuda.is_available()
    new_device = torch.device("cuda:0" if use_gpu else "cpu")

    if verbose:
        print(f"[TorchDR] Using device: {new_device}.")

    if isinstance(x, torch.Tensor):
        input_backend = "torch"
        input_device = x.device
        x_ = x.to(new_device) if input_device != new_device else x

    elif isinstance(x, np.ndarray):
        input_backend = "numpy"
        input_device = "cpu"
        x_ = torch.from_numpy(x).to(new_device)  # memory efficient

    else:
        raise ValueError(f"Unsupported type {type(x)}.")

    if return_backend_device:
        return x_, input_backend, input_device
    else:
        return x_


def torch_to_backend(x, backend="torch", device="cpu"):
    x = x.to(device=device)
    return x.numpy() if backend == "numpy" else x


def keops_unsqueeze(arg):
    """
    Apply unsqueeze(-1) to an input vector or batched vector.
    Then converts it to a KeOps lazy tensor.
    """
    if arg.ndim == 1:  # arg is a vector
        return LazyTensor(arg.unsqueeze(-1), 0)
    elif arg.ndim == 2:  # arg is a batched vector
        return LazyTensor(
            arg.unsqueeze(-1).unsqueeze(-1)
        )  # specifying to KeOps that we have a batch dimension
    else:
        raise ValueError("Unsupported input shape for keops_unsqueeze function.")


def wrap_vectors(func):
    """
    If the cost matrix C is a lazy tensor, convert all other input tensors to KeOps
    lazy tensors while applying unsqueeze(-1).
    These tensors should be vectors or batched vectors.
    """

    @functools.wraps(func)
    def wrapper(C, *args, **kwargs):
        use_keops = isinstance(C, LazyTensor)

        unsqueeze = lambda arg: keops_unsqueeze(arg) if use_keops else arg.unsqueeze(-1)

        args = [
            (unsqueeze(arg) if isinstance(arg, torch.Tensor) else arg) for arg in args
        ]
        kwargs = {
            key: (unsqueeze(value) if isinstance(value, torch.Tensor) else value)
            for key, value in kwargs.items()
        }
        return func(C, *args, **kwargs)

    return wrapper


def sum_all_axis(func):
    """
    Sum the output matrix over all axis.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        assert isinstance(output, torch.Tensor) or isinstance(
            output, LazyTensor
        ), "sum_all_axis can only be applied to a tensor or lazy tensor."
        return output.sum(1).sum()  # for compatibility with KeOps

    return wrapper


def handle_backend(func):
    """
    Convert input to torch and device specified by self.
    Then, convert the output to the input backend and device.
    """

    @functools.wraps(func)
    def wrapper(self, X):
        X_, input_backend, input_device = to_torch(
            X, device=self.device, verbose=False, return_backend_device=True
        )
        output = func(self, X_).detach()
        return torch_to_backend(output, backend=input_backend, device=input_device)

    return wrapper
