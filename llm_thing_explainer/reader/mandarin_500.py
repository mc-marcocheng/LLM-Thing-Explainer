from os import PathLike
from pathlib import Path

from . import register_reader


@register_reader("mandarin 500 chars")
def read_mandarin_500(file_path: PathLike = Path(__file__).parent / "mandarin_500.txt") -> list[str]:
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read().strip()
    words = list(content)
    return words

@register_reader("mandarin 250 chars")
def read_mandarin_250(file_path: PathLike = Path(__file__).parent / "mandarin_500.txt") -> list[str]:
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read().strip()
    words = list(content[:250] + content[500:])
    return words
