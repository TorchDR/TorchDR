# -*- coding: utf-8 -*-
"""
Spaces and associated metrics
"""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import torch
from abc import abstractmethod


class Geometry():
    @abstractmethod
    def pairwise_distances(self, X):
        pass


class Euclidean(Geometry):
    def pairwise_distances(self, X):
        r"""Computes pairwise Euclidean distances matrix."""
        return torch.cdist(X, X, p=2)
