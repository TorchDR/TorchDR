# -*- coding: utf-8 -*-
"""Dataset base classes."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License


from pathlib import Path, PosixPath
from typing import Union
from abc import abstractmethod
import pickle
from sklearn.datasets import fetch_openml
import time

from .utils import download_from_url, load_from_local_path, load_from_url


class Dataset:
    def __init__(
        self,
        save_path: Union[str, PosixPath],
        download: bool = True,
        verbose: bool = False,
    ):
        self.save_path = Path(save_path) / (
            self.__class__.__name__ + ".pkl"
        )  # append name to the save path
        self.download = download
        self.verbose = verbose

    def load(self):
        if self.verbose:
            print(f"[TorchDR] Loading the dataset {self.__class__.__name__}.")

        if self.download:
            if not self.save_path.exists():
                start_time = time.time()
                self._download()
                elapsed_time = time.time() - start_time

                if self.verbose:
                    print(
                        f"[TorchDR] Downloaded the dataset {self.__class__.__name__} "
                        f"in {elapsed_time:.2f} seconds."
                    )
            data = load_from_local_path(self.save_path)
        else:
            data = self._load_without_download()

        return self._create_features_labels(data)

    @abstractmethod
    def _download(self):
        raise NotImplementedError

    @abstractmethod
    def _load_without_download(self):
        raise NotImplementedError

    def _create_features_labels(self, data):
        return data


class URLDataset(Dataset):
    def __init__(
        self,
        save_path: Union[str, PosixPath],
        download: bool = True,
        verbose: bool = False,
    ):
        super().__init__(save_path, download, verbose)

    @property
    def url(self):
        raise NotImplementedError

    def _download(self):
        download_from_url(self.url, self.save_path)

    def _load_without_download(self):
        return load_from_url(self.url)


class OpenMLDataset(Dataset):
    def __init__(
        self,
        save_path: Union[str, PosixPath],
        download: bool = True,
        verbose: bool = False,
        dataset_name: str = None,
    ):
        super().__init__(save_path=save_path, download=download, verbose=verbose)
        self.dataset_name = dataset_name

    def _load_from_openml(self):
        dataset = fetch_openml(self.dataset_name, cache=True, as_frame=False)
        features = dataset.data.astype("float32")
        labels = dataset.target.astype("int64")
        return features, labels

    def _download(self):
        dataset = self._load_from_openml()
        with open(self.save_path, "wb") as f:
            pickle.dump(dataset, f)

    def _load_without_download(self):
        dataset = self._load_from_openml()
        return dataset
