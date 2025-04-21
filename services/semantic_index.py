import faiss
import numpy as np
import json
from typing import List, Tuple

def load_tag_index(index_path: str, metadata_path: str) -> Tuple[faiss.IndexFlatL2, List[dict]]:
    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return index, metadata

def query_tag_index(embedding: List[float], index: faiss.IndexFlatL2, metadata: List[dict], k: int = 3) -> List[dict]:
    embedding_np = np.array(embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(embedding_np, k)
    return [metadata[i] for i in indices[0]]
