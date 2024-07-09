# -*- coding: utf-8 -*-
"""
Robust handling of pykeops as optional dependency
"""

# Author: Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause License

try:
    import pykeops
    from pykeops.torch import LazyTensor
    LazyTensorType = LazyTensor
except ImportError:
    pykeops = False  # pykeops is not installed
    LazyTensor = None  # pykeops is not installed
    LazyTensorType = type(None)


def is_lazy_tensor(arg):
    r"""
    Returns True if the input is a KeOps lazy tensor.
    """
    if not pykeops:
        return False
    return isinstance(arg, LazyTensor)
