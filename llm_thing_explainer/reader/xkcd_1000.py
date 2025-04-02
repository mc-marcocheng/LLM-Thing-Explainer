from os import PathLike
from pathlib import Path

from . import register_reader


@register_reader("xkcd 1000 words")
def read_xkcd_1000(file_path: PathLike = Path(__file__).parent / "xkcd_1000.txt") -> list[str]:
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read().strip()
    words = content.split('|')
    return words
