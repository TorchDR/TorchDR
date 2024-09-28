import gzip
import pickle
import requests


def load_from_url(url, chunk_size=8192):
    """Download a gzip-compressed and pickled file from a URL and load the data.

    This function fetches data from the provided URL, decompresses the file
    using gzip, and unpickles the content to return the Python object.

    Parameters
    ----------
    url : str
        The URL of the gzip-compressed and pickled file to be downloaded.
    chunk_size : int, optional
        The size of the chunks in which to stream the data, by default 8192 (8 KB).

    Returns
    -------
    object or None
        The unpickled Python object if successful, or `None` if an error occurs.

    Raises
    ------
    requests.exceptions.RequestException
        If there is an issue with the network request
        (e.g., connection error or invalid URL).
    pickle.UnpicklingError
        If the data cannot be unpickled due to corruption or format issues.
    gzip.BadGzipFile
        If the downloaded file is not a valid gzip file or is corrupted.

    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        buffer = io.BytesIO()  # buffer to hold the streamed content

        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                buffer.write(chunk)

        buffer.seek(0)  # reset buffer to the start

        with gzip.open(buffer, "rb") as f:
            data = pickle.load(f)

        return data

    except requests.exceptions.RequestException as e:
        print(f"[TorchDR] Error downloading the file: {e}")
        return None

    except (pickle.UnpicklingError, gzip.BadGzipFile) as e:
        print(f"[TorchDR] Error unpickling or decompressing the data: {e}")
        return None


def download_from_url(url, save_path, chunk_size=8192):
    """Download a file from a URL and save it to the specified path.

    Parameters
    ----------
    url : str
        The URL of the file to be downloaded.
    save_path : str
        The local path where the downloaded file will be saved.
    chunk_size : int, optional
        The size of the chunks in which to stream the data, by default 8192 (8 KB).

    Returns
    -------
    str or None
        The path to the saved file if successful, or `None` if an error occurs.

    Raises
    ------
    requests.exceptions.RequestException
        If there is an issue with the network request
        (e.g., connection error or invalid URL).

    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        return save_path

    except requests.exceptions.RequestException as e:
        print(f"[TorchDR] Error downloading the file: {e}")
        return None


def load_from_local_path(save_path):
    """Load and unpickle data from a gzip-compressed file.

    Parameters
    ----------
    save_path : str
        The local path of the gzip-compressed and pickled file to be loaded.

    Returns
    -------
    object or None
        The unpickled Python object if successful, or `None` if an error occurs.

    Raises
    ------
    pickle.UnpicklingError
        If the data cannot be unpickled due to corruption or format issues.
    gzip.BadGzipFile
        If the file is not a valid gzip file or is corrupted.
    """
    try:
        # Open the file, decompress it, and load the pickled data
        with gzip.open(save_path, "rb") as f:
            data = pickle.load(f)

        return data

    except (pickle.UnpicklingError, gzip.BadGzipFile) as e:
        print(f"[TorchDR] Error unpickling or decompressing the file: {e}")
        return None
