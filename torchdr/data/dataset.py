from pathlib import Path, PosixPath
from typing import Union
import time

from .utils import load_from_url, download_from_url, load_from_local_path


class Dataset:
    def __init__(
        self,
        save_path: Union[str, PosixPath],
        download: bool = True,
        verbose: bool = False,
    ):
        self.save_path = (
            Path(save_path) / self.__class__.__name__
        )  # append name to the save path
        self.download = download

    @abstractmethod
    def load(self):
        raise NotImplementedError


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

    def load(self):
        start_time = time.time()

        # Load data
        if self.save_path.exists():
            with_dowlnoad = "without"
            data = load_from_local_path(self.save_path)
        else:
            with_download = "with"
            download_from_url(self.url, self.save_path)
            data = load_from_local_path(self.save_path)

        elapsed_time = time.time() - start_time

        if self.verbose:
            print(
                f"[TorchDR] Time taken to load the dataset {with_dowlnoad} download: "
                f"{elapsed_time:.2f} seconds."
            )

        return data
