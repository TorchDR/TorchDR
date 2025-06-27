"""Useful wrappers for dealing with backends and devices."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import functools
import torch

from .keops import LazyTensor, is_lazy_tensor, pykeops
from .validation import check_array


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
def to_torch(x, device="auto", return_backend_device=False, **check_array_kwargs):
    """Convert input to torch tensor and specified device while performing some checks.

    If device="auto", the device is set to the device of the input x.
    """
    if isinstance(x, torch.Tensor):
        input_backend = "torch"
        input_device = x.device
    else:
        input_backend = "numpy"
        input_device = "cpu"

    if device == "auto":
        target_device = input_device
    else:
        target_device = device

    x_ = check_array(x, device=target_device, **check_array_kwargs)

    if torch.is_complex(x_):
        raise ValueError("[TorchDR] ERROR : complex tensors are not supported.")
    if not torch.isfinite(x_).all():
        raise ValueError("[TorchDR] ERROR : input contains infinite values.")

    if not x_.dtype.is_floating_point:
        x_ = x_.float()

    if return_backend_device:
        return x_, input_backend, input_device
    else:
        return x_


def torch_to_backend(x, backend="torch", device="cpu"):
    """Convert a torch tensor to specified backend and device."""
    if not isinstance(x, torch.Tensor):
        return x  # Return as is if not a tensor

    if backend == "numpy":
        return x.detach().cpu().numpy()
    else:
        return x.to(device=device)


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
        raise ValueError(
            "[TorchDR] ERROR : Unsupported input shape for keops_unsqueeze function."
        )


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


def handle_type(_func=None, *, set_device=True, **check_array_kwargs):
    """
    Convert input to torch and optionally set device specified by self.

    Then convert the output to the input backend and device.

    Parameters
    ----------
    _func : callable, optional
        The function to be wrapped.
    set_device : bool, default=True
        If True, set the device to self.device if it is not None.
    **check_array_kwargs : dict
        Keyword arguments to be passed to the check_array function.
    """

    def decorator_handle_type(func):
        @functools.wraps(func)
        def wrapper(self, X, *args, **kwargs):
            # Use self.device if set_device is True, else leave device unset (None)
            device = self.device if set_device else "auto"
            X_, input_backend, input_device = to_torch(
                X,
                device=device,
                return_backend_device=True,
                **check_array_kwargs,
            )
            output = func(self, X_, *args, **kwargs)
            return torch_to_backend(output, backend=input_backend, device=input_device)

        return wrapper

    # Support both @handle_type and @handle_type(...)
    if _func is None:
        return decorator_handle_type
    else:
        return decorator_handle_type(_func)


def handle_keops(func):
    """Set the backend_ attribute to 'keops' if an OutOfMemoryError is encountered.

    If backend is set to 'keops', backend_ is also set to 'keops' and nothing is done.
    Otherwise, the function is called and if an OutOfMemoryError is encountered,
    backend_ is set to 'keops' and the function is called again.
    """

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
