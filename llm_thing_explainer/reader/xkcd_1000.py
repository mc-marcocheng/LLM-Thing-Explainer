from os import PathLike
from pathlib import Path


def read_xkcd_1000(file_path: PathLike = Path(__file__).parent / "xkcd_1000.txt") -> list[str]:
    with open(file_path, 'r') as file:
        content = file.read().strip()
    words = content.split('|')
    return words
