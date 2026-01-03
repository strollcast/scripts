import sys
from pathlib import Path
from urllib.request import urlretrieve


def download(url: str):
    """Download script from R2.

    Example argument
      https://released.strollcast.com/episodes/chen-2023-punica_multi_tenant/script.md
    """
    folder = Path(url.split("/")[-2])
    if not folder.exists():
        folder.mkdir()
    dest = folder / "script.md"
    if dest.exists():
        print(f"File {dest} already exists.")
        return
    urlretrieve(url, dest)
    if dest.exists():
        print(f"Downloaded {dest}.")
    else:

        print(f"Error downloading {url}")


if __name__ == "__main__":
    download(sys.argv[1])
