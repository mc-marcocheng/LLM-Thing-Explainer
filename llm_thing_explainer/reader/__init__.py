import importlib
import pathlib

WORD_READERS = {}

def register_reader(reader_name):
    def decorator(func):
        WORD_READERS[reader_name] = func
        return func
    return decorator

# Discover and import all reader modules in the current directory
for module_file in pathlib.Path(__file__).parent.glob("*.py"):
    module_name = module_file.stem
    if module_name != "__init__":
        importlib.import_module(f".{module_name}", package=__package__)
