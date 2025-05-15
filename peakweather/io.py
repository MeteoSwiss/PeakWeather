import os
import tarfile
import urllib.request
import zipfile
from types import ModuleType
from typing import Optional

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str,
                 folder: str,
                 filename: Optional[str] = None) -> str:
    r"""Downloads the content of a URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        filename (string, optional): The filename. If :obj:`None`, it is inferred from
            the url.

    Returns:
        string: The path to the downloaded file.

    Raises:
        FileExistsError: If the file already exists in the specified folder.
    """
    if filename is None:
        filename = url.rpartition('/')[2].split('?')[0]
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        raise FileExistsError(f"File {path} already exists. "
                              "Please remove it before downloading again.")

    os.makedirs(folder, exist_ok=True)
    with DownloadProgressBar(unit='B',
                             unit_scale=True,
                             miniters=1,
                             desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)
    return path


def extract_zip(path: str, folder: str):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the zip archive.
        folder (string): The folder.
    """
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def extract_tar(path: str, folder: str):
    r"""Extracts a tar (or tar.gz) archive to a specific folder.

    Args:
        path (string): The path to the tar(gz) archive.
        folder (string): The destination folder.
    """
    with tarfile.open(path, 'r') as tar:
        for member in tqdm(iterable=tar.getmembers(),
                           total=len(tar.getmembers())):
            tar.extract(member=member, path=folder)


def import_xarray() -> ModuleType:
    """Attempts to import the 'xarray' module and return it.

    Returns:
        types.ModuleType: The imported 'xarray' module.

    Raises:
        ModuleNotFoundError: If 'xarray' is not installed.
    """
    try:
        import zarr
    except ImportError as e:
        raise ModuleNotFoundError(
            "The 'zarr' library is required for this functionality. "
            "You can install it via pip:\n\n    pip install 'zarr<3.0.0'\n"
        ) from e
    try:
        import xarray
        return xarray
    except ImportError as e:
        raise ModuleNotFoundError(
            "The 'xarray' library is required for this functionality. "
            "You can install it via pip:\n\n    pip install xarray\n"
        ) from e
