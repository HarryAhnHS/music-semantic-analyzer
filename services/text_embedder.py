import os
import json
import numpy as np
import faiss
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer

class TextEmbeddingIndex:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = str(index_path)
        self.metadata_path = str(metadata_path)
        

        # Ensure parent directories exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # Initialize or load FAISS index
        if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
            self.index = faiss.read_index(self.index_path)
            print(f"[FAISS] ✅ Loaded text index from {self.index_path}")
        else:
            self.index = faiss.IndexFlatL2(384)  # all-MiniLM-L6-v2 embedding size
            print(f"[FAISS] 🆕 Created new text index at {self.index_path}")

        # Initialize or load metadata
        if os.path.exists(metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    content = f.read().strip()
                    self.metadata = json.loads(content) if content else []
                print(f"[META] ✅ Loaded {len(self.metadata)} metadata entries.")
            except Exception as e:
                print(f"[META] ⚠️ Failed to load metadata ({e}). Initializing empty.")
                self.metadata = []
        else:
            self.metadata = []
            print(f"[META] 🆕 Created new metadata list for {self.metadata_path}")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text_blob(self, text_blob: str) -> List[float]:
        return self.model.encode(text_blob.strip()).astype(np.float32).tolist()

    def add_entry(self, text_blob: str, metadata: Optional[dict] = None) -> List[float]:
        embedding = np.array([self.embed_text_blob(text_blob)], dtype=np.float32)
        self.index.add(embedding)
        self.metadata.append(metadata or {})
        return embedding[0].tolist()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"[SAVE] Index and metadata saved.")

    def query(self, query_text: str, k: int = 5) -> List[dict]:
        embedding = np.array([self.embed_text_blob(query_text)], dtype=np.float32)
        distances, indices = self.index.search(embedding, k)
        return [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
    
    def generate_text_blob(self, entry: dict) -> str:
        """
        Generates a freeform, natural-language-style text blob from semantic metadata.
        Ideal for use in text embeddings and LLM similarity comparisons.
        """
        metadata = entry.get("metadata", {})
        genre = metadata.get("genre", "").strip()
        track_type = metadata.get("track_type", "").strip()

        tags = entry.get("tags", [])
        summary = entry.get("summary", "").strip()
        stem_tags = entry.get("stem_tags", {})
        stem_summaries = entry.get("stem_summaries", {})

        # Compose a natural-sounding description
        parts = []

        if summary:
            parts.append(summary)

        if genre:
            parts.append(f"It falls within the {genre} genre.")

        if tags:
            parts.append(f"The track has characteristics such as {', '.join(tags)}.")

        if track_type:
            parts.append(f"This is a {track_type}.")

        for stem in ["vocals", "drums", "bass", "other"]:
            t = stem_tags.get(stem, [])
            s = stem_summaries.get(stem, "")

            if t and t != ["unknown"]:
                parts.append(f"The {stem} are described as {', '.join(t)}.")
            if s and "Unable to generate summary." not in s:
                parts.append(s.strip())

        final_text = " ".join(parts).replace("\n", " ").strip()
        print("✅ Natural text blob to encode:\n", final_text)
        return final_text