from os import PathLike
from pathlib import Path

from . import register_reader


@register_reader("cat sounds")
def read_cat_sounds(file_path: PathLike = Path(__file__).parent / "cat.txt") -> list[str]:
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read().strip()
    words = content.split('|')
    return words
