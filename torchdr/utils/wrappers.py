# -*- coding: utf-8 -*-
"""Useful wrappers for dealing with backends and devices."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import functools
import torch
import numpy as np
from .keops import LazyTensor, is_lazy_tensor
from sklearn.utils.validation import check_array


def output_contiguous(func):
    """Convert all output torch tensors to contiguous."""

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


@output_contiguous
def to_torch(x, device="auto", return_backend_device=False):
    """Convert input to torch tensor and specified device while performing some checks.

    If device="auto", the device is set to the device of the input x.
    """
    gpu_required = (
        device in ["cuda", "cuda:0", "gpu", None]
    ) and torch.cuda.is_available()

    new_device = torch.device("cuda:0" if gpu_required else "cpu")

    if isinstance(x, torch.Tensor):
        if torch.is_complex(x):
            raise ValueError("[TorchDR] ERROR : complex tensors are not supported.")
        if not torch.isfinite(x).all():
            raise ValueError("[TorchDR] ERROR : input contains infinite values.")

        input_backend = "torch"
        input_device = x.device

        if device == "auto" or input_device == new_device:
            x_ = x
        else:
            x_ = x.to(new_device)

    else:
        # check sparsity and if it contains only finite values
        if x.ndim == 2:
            x = check_array(x, accept_sparse=False)
        input_backend = "numpy"
        input_device = "cpu"

        if np.iscomplex(x).any():
            raise ValueError("[TorchDR] ERROR : complex arrays are not supported.")

        x_ = torch.from_numpy(x.copy()).to(new_device)  # memory efficient

    if not x_.dtype.is_floating_point:
        x_ = x_.float()  # KeOps does not support int

    if return_backend_device:
        return x_, input_backend, input_device
    else:
        return x_


def torch_to_backend(x, backend="torch", device="cpu"):
    """Convert a torch tensor to specified backend and device."""
    x = x.to(device=device)
    return x.numpy() if backend == "numpy" else x


def keops_unsqueeze(arg):
    """Apply unsqueeze(-1) to an input vector or batched vector.

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
    """Unsqueeze(-1) all input tensors except the cost matrix C.

    If C is a lazy tensor, converts all tensors to KeOps lazy tensors.
    These tensors should be vectors or batched vectors.
    """

    @functools.wraps(func)
    def wrapper(C, *args, **kwargs):
        use_keops = is_lazy_tensor(C)

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


def sum_output(func):
    """Sum the output over all axis if the tensor has 2 dimensions.

    Sum the output over all axis except the batch axis if the tensor has 3 dimensions.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        ndim_output = len(output.shape)

        if not (isinstance(output, torch.Tensor) or is_lazy_tensor(output)):
            raise ValueError(
                "[TorchDR] ERROR : sum_all_axis_except_batch can only be applied "
                "to a torch.Tensor or pykeops.torch.LazyTensor."
            )
        elif ndim_output == 2:
            return output.sum(1).sum(0)
        elif ndim_output == 3:
            return output.sum(2).sum(1)
        else:
            raise ValueError(
                "[TorchDR] ERROR : Unsupported input shape for "
                "sum_all_axis_except_batch function."
            )

    return wrapper


def handle_backend(func):
    """Convert input to torch and device specified by self.

    Then convert the output to the input backend and device.
    """

    @functools.wraps(func)
    def wrapper(self, X, *args, **kwargs):
        X_, input_backend, input_device = to_torch(
            X, device=self.device, return_backend_device=True
        )
        output = func(self, X_, *args, **kwargs).detach()
        return torch_to_backend(output, backend=input_backend, device=input_device)

    return wrapper
