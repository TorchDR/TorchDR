"""Useful wrappers for dealing with backends and devices."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import functools
import torch
from torch.utils.data import DataLoader

from .keops import LazyTensor, is_lazy_tensor
from .validation import validate_tensor

import warnings

try:
    import pandas as pd
except ImportError:
    pd = None


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
def to_torch(x, return_backend_device=False):
    """Convert input to torch tensor without changing device.

    Handles conversion from numpy arrays, pandas DataFrames, lists, and DataLoaders.
    Preserves the original device and tracks the backend for later restoration.
    """
    if pd is not None and isinstance(x, pd.DataFrame):
        x = x.values

    if isinstance(x, DataLoader):
        # DataLoader: pass through without conversion
        input_backend = "dataloader"
        input_device = "cpu"
        x_ = x
    elif isinstance(x, torch.Tensor):
        input_backend = "torch"
        input_device = x.device
        x_ = x
    else:
        input_backend = "numpy"
        input_device = "cpu"
        try:
            x_ = torch.as_tensor(x)
        except (TypeError, ValueError):
            raise ValueError("Input could not be converted to a tensor.")

    # Convert to float if needed (skip for DataLoader)
    if not isinstance(x_, DataLoader) and not x_.dtype.is_floating_point:
        x_ = x_.float()

    if return_backend_device:
        return x_, input_backend, input_device
    else:
        return x_


def restore_original_format(x, backend="torch", device="cpu"):
    """Restore output to original format (numpy array or torch tensor with original device)."""
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


def handle_input_output(
    _func=None,
    *,
    accept_sparse=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    ensure_2d=True,
    **check_array_kwargs,
):
    """
    Handle input conversion to torch and output conversion back to original format.

    Converts input to torch tensor for processing while preserving the original
    device, then converts output back to the original backend and device.

    Parameters
    ----------
    _func : callable, optional
        The function to be wrapped.
    accept_sparse : bool, default=False
        Whether to accept sparse matrices.
    ensure_min_samples : int, default=2
        Minimum number of samples required.
    ensure_min_features : int, default=1
        Minimum number of features required.
    ensure_2d : bool, default=True
        Whether to ensure 2D input.
    **check_array_kwargs : dict
        Additional keyword arguments to be passed to the validate_tensor function.
    """

    def decorator_handle_input_output(func):
        @functools.wraps(func)
        def wrapper(self, X, *args, **kwargs):
            # Convert to torch tensor
            X_, input_backend, input_device = to_torch(
                X,
                return_backend_device=True,
            )

            # Validate the tensor
            X_ = validate_tensor(
                X_,
                accept_sparse=accept_sparse,
                ensure_min_samples=ensure_min_samples,
                ensure_min_features=ensure_min_features,
                ensure_2d=ensure_2d,
                **check_array_kwargs,
            )

            output = func(self, X_, *args, **kwargs)
            return restore_original_format(
                output, backend=input_backend, device=input_device
            )

        return wrapper

    # Support both @handle_input_output and @handle_input_output(...)
    if _func is None:
        return decorator_handle_input_output
    else:
        return decorator_handle_input_output(_func)


def compile_if_requested(func):
    """Decorator to conditionally compile a function with torch.compile.

    The compilation is triggered based on a 'compile' flag.
    For class methods, it checks for a `self.compile` attribute.
    For standalone functions, it checks for a `compile` keyword argument.

    The compiled function is cached for subsequent calls.
    """
    compiled_funcs = {}  # Cache for compiled functions

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if we should compile
        should_compile = False
        is_method = False
        if args and hasattr(args[0], "compile"):
            self = args[0]
            should_compile = getattr(self, "compile", False)
            is_method = True
        elif "compile" in kwargs:
            should_compile = kwargs["compile"]

        if not should_compile:
            return func(*args, **kwargs)

        # Create a unique key for the compiled function
        # For methods, key on the instance id to recompile for different instances
        # For functions, key on the function itself
        key = (id(self), func) if is_method else func

        if key in compiled_funcs:
            return compiled_funcs[key](*args, **kwargs)

        try:
            compiled_func = torch.compile(func)
            compiled_funcs[key] = compiled_func
        except Exception as e:
            msg = (
                f"Could not compile {func.__name__} with torch.compile. "
                f"Falling back to eager execution. Reason: {e}"
            )

            # For methods, try to use a logger
            if is_method and hasattr(self, "logger") and self.logger is not None:
                self.logger.warning(msg)
            else:
                warnings.warn(f"[TorchDR] WARNING: {msg}", UserWarning)

            # Cache the original function to avoid recompilation attempts
            compiled_funcs[key] = func
            return func(*args, **kwargs)

        return compiled_funcs[key](*args, **kwargs)

    return wrapper
