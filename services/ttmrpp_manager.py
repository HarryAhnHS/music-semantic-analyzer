from functools import lru_cache
from services.ttmrpp_wrapper import TTMRPPWrapper

@lru_cache(maxsize=8)
def get_ttmr(index_path: str, metadata_path: str, read_only: bool = False) -> TTMRPPWrapper:
    return TTMRPPWrapper(index_path, metadata_path, read_only)