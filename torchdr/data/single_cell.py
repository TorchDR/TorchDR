from .dataset import URLDataset
from .utils import load_from_local_path, download_from_url, load_from_url


class Macosko2015(URLDataset):
    def __init__(self, save_path: Union[str, PosixPath], download: bool = True):
        super().__init__(save_path, download)

    @property
    def url(self):
        return "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz"

    def download(self):
        download_from_url(self.url, self.save_path)


class Zheng2017(Dataset):
    def __init__(self, save_path: Union[str, PosixPath], download: bool = True):
        super().__init__(save_path, download)

    @property
    def url(self):
        return "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"

    def download(self):
        download_from_url(self.url, self.save_path)
