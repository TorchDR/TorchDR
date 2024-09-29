# -*- coding: utf-8 -*-
"""Single-cell datasets."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from pathlib import PosixPath
from typing import Union

from .dataset import URLDataset


class Macosko2015(URLDataset):
    def __init__(
        self,
        save_path: Union[str, PosixPath],
        download: bool = True,
        verbose: bool = False,
    ):
        super().__init__(save_path=save_path, download=download, verbose=verbose)

    @property
    def url(self):
        return "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz"

    def _create_features_labels(self, data):
        features = data["pca_50"].astype("float32")
        labels = data["CellType1"].astype(str)
        return features, labels


class Zheng2017(URLDataset):
    def __init__(
        self,
        save_path: Union[str, PosixPath],
        download: bool = True,
        verbose: bool = False,
    ):
        super().__init__(save_path=save_path, download=download, verbose=verbose)

    @property
    def url(self):
        return "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"

    def _create_features_labels(self, data):
        features = data["pca_50"].astype("float32")
        labels = data["CellType1"].astype(str)
        return features, labels
