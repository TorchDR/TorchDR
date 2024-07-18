# -*- coding: utf-8 -*-
"""Robust handling of pykeops as optional dependency."""

# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause License

try:
    import pykeops
    from pykeops.torch import LazyTensor

    LazyTensorType = LazyTensor

except ImportError:  # pykeops is not installed
    pykeops = False
    LazyTensor = None
    LazyTensorType = type(None)


def is_lazy_tensor(arg):
    r"""Return True if the input is a KeOps lazy tensor."""
    if not pykeops:
        return False
    return isinstance(arg, LazyTensor)
