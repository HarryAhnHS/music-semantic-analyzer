import torch
torch.set_num_threads(1)
import librosa
import numpy as np
import laion_clap
import os
import faiss
import json
from typing import Optional

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

class CLAPWrapper:
    def __init__(self, faiss_path=None, metadata_path=None):
        print("[CLAP init] Selecting device...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLAP init] Device set to {self.device}")

        self.model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel='HTSAT-base'
        )
        print("[CLAP init] Model instantiated, loading checkpoint...")
        self.model.load_ckpt("checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt")
        print("[CLAP init] Checkpoint loaded.")

        self.faiss_path = faiss_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []

        if faiss_path:
            os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
            if os.path.exists(faiss_path):
                print(f"[faiss] Loading existing index from {faiss_path}")
                self.index = faiss.read_index(str(faiss_path))
            else:
                print(f"[faiss] Creating new FAISS index at {faiss_path}")
                self.index = faiss.IndexFlatL2(512)

        if metadata_path:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        content = f.read().strip()
                        self.metadata = json.loads(content) if content else []
                except Exception as e:
                    print(f"[meta] Failed to load metadata: {e}")
                    self.metadata = []
            else:
                self.metadata = []

    def get_embedding(self, file_path: str) -> list[float]:
        audio_data, _ = librosa.load(file_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)
        audio_data = float32_to_int16(audio_data)
        audio_data = int16_to_float32(audio_data)
        audio_tensor = torch.from_numpy(audio_data).float().to(self.device)

        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Empty or unreadable audio file.")

        with torch.no_grad():
            embedding = self.model.get_audio_embedding_from_data(audio_tensor, use_tensor=True)

        return embedding.squeeze().cpu().numpy().tolist()

    def add_embedding_to_index(self, embedding: list[float], metadata: Optional[dict] = None):
        if self.index:
            self.index.add(np.array([embedding], dtype="float32"))
        if metadata is not None:
            self.metadata.append(metadata)
        else:
            self.metadata.append({})
        return embedding

    def save_index(self):
        if self.index and self.faiss_path:
            faiss.write_index(self.index, str(self.faiss_path))
        if self.metadata_path:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

    def query_neighbors(self, embedding: list[float], k: int = 3) -> list[tuple[int, float]]:
        if self.index is None:
            raise ValueError("No FAISS index loaded.")
        embedding_np = np.array(embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(embedding_np, k)
        return list(zip(indices[0], distances[0]))

    def query_neighbors_with_tagging_metadata(self, embedding: list[float], k: int = 3) -> list[dict]:
        if self.metadata is None:
            raise ValueError("No metadata loaded. Pass `metadata_path` to the constructor.")
        neighbor_info = self.query_neighbors(embedding, k)
        return [self.metadata[i] for i, _ in neighbor_info if i < len(self.metadata)]
