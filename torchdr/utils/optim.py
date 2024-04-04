# -*- coding: utf-8 -*-
"""
Useful tools for optimization problems
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch

OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "NAdam": torch.optim.NAdam,
}
