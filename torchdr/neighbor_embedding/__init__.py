# -*- coding: utf-8 -*-
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from .sne import SNE
from .tsne import TSNE
from .ncsne import InfoTSNE
from .umap import UMAP

__all__ = ["SNE", "TSNE", "InfoTSNE", "UMAP"]
