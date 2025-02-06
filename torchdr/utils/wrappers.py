"""Useful wrappers for dealing with backends and devices."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import functools

import numpy as np
import torch
from sklearn.utils.validation import check_array

from .keops import LazyTensor, is_lazy_tensor, pykeops


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
    if isinstance(x, torch.Tensor):
        if torch.is_complex(x):
            raise ValueError("[TorchDR] ERROR : complex tensors are not supported.")
        if not torch.isfinite(x).all():
            raise ValueError("[TorchDR] ERROR : input contains infinite values.")

        input_backend = "torch"
        input_device = x.device

        if device == "auto":
            x_ = x
        else:
            x_ = x.to(device)

    else:
        # check sparsity and if it contains only finite values
        if x.ndim == 2:
            x = check_array(x, accept_sparse=False)
        input_backend = "numpy"
        input_device = "cpu"

        if np.iscomplex(x).any():
            raise ValueError("[TorchDR] ERROR : complex arrays are not supported.")

        x_ = torch.from_numpy(x.copy()).to(
            torch.device("cpu") if device == "auto" else device
        )

    if not x_.dtype.is_floating_point:
        x_ = x_.float()

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

        def unsqueeze(arg):
            return keops_unsqueeze(arg) if use_keops else arg.unsqueeze(-1)

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
                f"[TorchDR] ERROR : {func.__name__} can only be applied "
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


def handle_type(_func=None, *, set_device=True):
    """
    Convert input to torch and optionally set device specified by self.

    Then convert the output to the input backend and device.

    Parameters
    ----------
    _func : callable, optional
        The function to be wrapped.
    set_device : bool, default=True
        If True, set the device to self.device if it is not None.
    """

    def decorator_handle_type(func):
        @functools.wraps(func)
        def wrapper(self, X, *args, **kwargs):
            # Use self.device if set_device is True, else leave device unset (None)
            device = self.device if set_device else "auto"
            X_, input_backend, input_device = to_torch(
                X, device=device, return_backend_device=True
            )
            output = func(self, X_, *args, **kwargs).detach()
            return torch_to_backend(output, backend=input_backend, device=input_device)

        return wrapper

    # Support both @handle_type and @handle_type(set_device=...)
    if _func is None:
        return decorator_handle_type
    else:
        return decorator_handle_type(_func)


def handle_keops(func):
    """Set the backend_ attribute to 'keops' if an OutOfMemoryError is encountered.

    If backend is set to 'keops', backend_ is also set to 'keops' and nothing is done.
    Otherwise, the function is called and if an OutOfMemoryError is encountered,
    backend_ is set to 'keops' and the function is called again.
    """  # noqa: RST306

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # if indices are provided, we do not use KeOps
        if kwargs.get("indices", None) is not None:
            return func(self, *args, **kwargs)

        if not hasattr(self, "backend_"):
            self.backend_ = self.backend
            if self.backend_ != "keops":
                try:
                    return func(self, *args, **kwargs)

                except torch.cuda.OutOfMemoryError:
                    print(
                        "[TorchDR] Out of memory encountered, setting backend to 'keops' "
                        f"for {self.__class__.__name__} object."
                    )
                    if not pykeops:
                        raise ValueError(
                            "[TorchDR] ERROR : pykeops is not installed. "
                            "To use `backend='keops'`, please run `pip install pykeops` "
                            "or `pip install torchdr[all]`. "
                        )
                    self.backend_ = "keops"

        return func(self, *args, **kwargs)

    return wrapper
