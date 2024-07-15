# -*- coding: utf-8 -*-
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from .sne import SNE
from .tsne import TSNE
from .ncsne import InfoTSNE
from .tsnekhorn import TSNEkhorn
from .largevis import LargeVis
from .umap import UMAP
from .base import NeighborEmbedding, SparseNeighborEmbedding, SampledNeighborEmbedding

__all__ = [
    "NeighborEmbedding",
    "SparseNeighborEmbedding",
    "SampledNeighborEmbedding",
    "SNE",
    "TSNE",
    "InfoTSNE",
    "TSNEkhorn",
    "LargeVis",
    "UMAP",
]
