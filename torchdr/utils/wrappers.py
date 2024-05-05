# -*- coding: utf-8 -*-
"""
Useful wrappers for dealing with backends and devices
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import functools
import itertools
import torch
import numpy as np
from pykeops.torch import LazyTensor


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


def wrap_vectors(func):
    """
    Reshape all input vectors from size (n) to size (n, 1).
    If any input is a lazy tensor, convert all input vectors to lazy tensors.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        use_keops = any(
            isinstance(arg, LazyTensor)
            for arg in itertools.chain(args, kwargs.values())
        )
        is_vector = lambda arg: isinstance(arg, torch.Tensor) and arg.ndim == 1
        unsqueeze = lambda arg: (
            LazyTensor(arg[:, None], 0) if use_keops else arg[:, None]
        )

        args = [unsqueeze(arg) if is_vector(arg) else arg for arg in args]
        kwargs = {
            key: (unsqueeze(value) if is_vector(value) else value)
            for key, value in kwargs.items()
        }
        return func(*args, **kwargs)

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
        ), "sum_all_axis can be applied to a tensor or lazy tensor."
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
        output = func(self, X_)
        return torch_to_backend(output, backend=input_backend, device=input_device)

    return wrapper
