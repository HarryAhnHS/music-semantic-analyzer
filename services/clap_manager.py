# services/clap_manager.py
from functools import lru_cache
from services.clap_wrapper import CLAPWrapper

@lru_cache(maxsize=8)
def get_clap(index_path: str, metadata_path: str) -> CLAPWrapper:
    return CLAPWrapper(index_path, metadata_path)