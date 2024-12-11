# -*- coding: utf-8 -*-
"""Images datasets."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from pathlib import PosixPath
from typing import Union

from .dataset import OpenMLDataset
from .utils import download_from_url


class MNIST(OpenMLDataset):
    def __init__(
        self,
        save_path: Union[str, PosixPath],
        download: bool = True,
        verbose: bool = False,
    ):
        super().__init__(
            save_path=save_path,
            download=download,
            verbose=verbose,
            dataset_name="mnist_784",
        )
