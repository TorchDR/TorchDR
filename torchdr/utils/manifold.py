# -*- coding: utf-8 -*-
"""
Robust handling of geoopt as optional dependency
"""

# Author: Nicolas Courty <ncourty@irisa.fr>
#
# License: BSD 3-Clause License

# from .optim import OPTIMIZERS
try:
    import geoopt

except Exception:  # geoopt is not installed
    geoopt = False


def is_geoopt_available():
    if not geoopt:
        raise ValueError(
            "[TorchDR] ERROR : geoopt is not installed. Please install it to use "
            "`manifold=true`."
        )
    else:
        return True
