# 👇 Add this early
import sys
import os
import torch
import numpy as np
import json
import faiss
from typing import Optional, List, Tuple
from external.music_text_representation_pp.mtrpp.utils.eval_utils import load_ttmr_pp
from external.music_text_representation_pp.mtrpp.utils.audio_utils import (
    int16_to_float32, float32_to_int16,
    load_audio, STR_CH_FIRST
)

from services.ttmrpp_singleton import TTMR_MODEL, TTMR_DEVICE


SR = 22050
N_SAMPLES = int(SR * 10)

from pathlib import Path

class TTMRPPWrapper:
    def __init__(
        self,
        app=None,
        variant: Optional[str] = None,
        faiss_path=None,
        metadata_path=None,
        read_only: bool = False,
        model_dir: str = "models/ttmrpp",
        model_type: str = "best",
        gpu: int = 0,
    ):
        self.device = TTMR_DEVICE
        self.model = TTMR_MODEL
        self.read_only = read_only
        self.index = None
        self.metadata = []

        # ✅ Check app.state if variant and app are provided
        if variant and app is not None and hasattr(app.state, "faiss_variants"):
            variants = app.state.faiss_variants
            if variant in variants:
                print(f"[TTMR] Using preloaded variant '{variant}' from app.state")
                self.index = variants[variant]["index"]
                self.metadata = variants[variant]["metadata"]
                return
            else:
                print(f"[TTMR] Variant '{variant}' not found in app.state.faiss_variants, falling back to file paths")

        # ✅ Legacy fallback: direct faiss_path + metadata_path
        self.faiss_path = faiss_path
        self.metadata_path = metadata_path

        if faiss_path:
            faiss_path = Path(faiss_path)
            os.makedirs(faiss_path.parent, exist_ok=True)
            if faiss_path.exists():
                print(f"[faiss] Loading index from {faiss_path}")
                self.index = faiss.read_index(
                    str(faiss_path),
                    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if read_only else 0
                )
            else:
                print(f"[faiss] Creating new FAISS index at {faiss_path}")
                self.index = faiss.IndexFlatL2(128)

        if metadata_path:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        content = f.read().strip()
                        self.metadata = json.loads(content) if content else []
                except Exception as e:
                    print(f"[meta] Failed to load metadata: {e}")
                    self.metadata = []
            else:
                self.metadata = []


    def _load_model(self, save_dir: str, model_type: str):
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, "best.pth")
        hparams_path = os.path.join(save_dir, "hparams.yaml")

        if not os.path.isfile(ckpt_path):
            print("⬇️ Downloading TTMR++ checkpoint...")
            torch.hub.download_url_to_file(
                'https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.pth', ckpt_path
            )
        if not os.path.isfile(hparams_path):
            print("⬇️ Downloading TTMR++ hparams config...")
            torch.hub.download_url_to_file(
                'https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.yaml', hparams_path
            )

        print("📦 Loading TTMR++ model...")
        model, _, _ = load_ttmr_pp(save_dir, model_types=model_type)
        return model

    def _load_wav_tensor(self, audio_path: str) -> torch.Tensor:
        audio, _ = load_audio(audio_path, STR_CH_FIRST, SR, downmix_to_mono=True)
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        audio = int16_to_float32(float32_to_int16(audio))
        ceil = int(audio.shape[-1] // N_SAMPLES)
        audio_tensor = torch.from_numpy(
            np.stack(np.split(audio[:ceil * N_SAMPLES], ceil)).astype('float32')
        )
        return audio_tensor

    def get_audio_embedding(self, audio_path: str) -> torch.Tensor:
        audio_tensor = self._load_wav_tensor(audio_path).to(self.device)
        with torch.no_grad():
            z_audio = self.model.audio_forward(audio_tensor)
        return z_audio.mean(0).detach().cpu().float()

    def get_text_embedding(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            z_text = self.model.text_forward([text])
        return z_text.squeeze(0).detach().cpu().float()

    def query_neighbors(self, embedding: list[float], k: int = 3) -> list[tuple[int, float]]:
        if self.index is None:
            raise ValueError("No FAISS index loaded.")
        embedding_np = np.array(embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(embedding_np, k)
        return list(zip(indices[0], distances[0]))

    def query_neighbors_with_metadata(self, embedding: list[float], k: int = 3) -> list[dict]:
        if self.metadata is None:
            raise ValueError("No metadata loaded. Pass `metadata_path` to the constructor.")
        neighbor_info = self.query_neighbors(embedding, k)
        return [self.metadata[i] for i, _ in neighbor_info if i < len(self.metadata)]

    def add_embedding_to_index(self, embedding: list[float], metadata: Optional[dict] = None):
        if self.read_only:
            raise RuntimeError("Cannot add to read-only index.")
        if self.index:
            self.index.add(np.array([embedding], dtype="float32"))
        self.metadata.append(metadata if metadata is not None else {})
        return embedding

    def save_index(self):
        if self.read_only:
            raise RuntimeError("Cannot save in read-only mode.")
        if self.index and self.faiss_path:
            faiss.write_index(self.index, str(self.faiss_path))
        if self.metadata_path:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
