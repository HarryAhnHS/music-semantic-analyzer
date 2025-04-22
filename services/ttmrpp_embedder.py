# services/ttmrpp_embedder.py

import os
import torch
import numpy as np
import sys

# Automatically add the TTMR++ repo to sys.path
TTMRPP_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "music-text-representation-pp"))
if TTMRPP_REPO not in sys.path:
    sys.path.append(TTMRPP_REPO)

# Now import TTMR++ modules
from mtrpp.utils.eval_utils import load_ttmr_pp
from mtrpp.utils.audio_utils import (
    int16_to_float32, float32_to_int16,
    load_audio, STR_CH_FIRST
)

SR = 22050
N_SAMPLES = int(SR * 10)

# Cached model (singleton)
_ttmr_model = None

def ensure_model_loaded(gpu=0):
    global _ttmr_model
    if _ttmr_model is None:
        print("🧠 Loading TTMR++ model...")
        save_dir = "models/ttmrpp"
        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = os.path.join(save_dir, "best.pth")
        hparams_path = os.path.join(save_dir, "hparams.yaml")

        if not os.path.isfile(ckpt_path):
            print("⬇️ Downloading model checkpoint...")
            torch.hub.download_url_to_file(
                'https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.pth', ckpt_path
            )
        if not os.path.isfile(hparams_path):
            print("⬇️ Downloading model config...")
            torch.hub.download_url_to_file(
                'https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.yaml', hparams_path
            )

        print("📦 Loading model weights from disk...")
        model, sr, duration = load_ttmr_pp(save_dir, model_types="best")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Moving model to {device}...")
        model = model.to(device).eval()

        _ttmr_model = model
        print("✅ Model ready.")
    return _ttmr_model



def load_wav_tensor(audio_path: str) -> torch.Tensor:
    audio, _ = load_audio(
        path=audio_path,
        ch_format=STR_CH_FIRST,
        sample_rate=SR,
        downmix_to_mono=True
    )
    if len(audio.shape) == 2:
        audio = audio.squeeze(0)
    audio = int16_to_float32(float32_to_int16(audio))
    ceil = int(audio.shape[-1] // N_SAMPLES)
    audio_tensor = torch.from_numpy(
        np.stack(np.split(audio[:ceil * N_SAMPLES], ceil)).astype('float32')
    )
    return audio_tensor


def get_ttmr_audio_embedding(audio_path: str, gpu: int = 0) -> torch.Tensor:
    model = ensure_model_loaded(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    audio = load_wav_tensor(audio_path).to(device)
    with torch.no_grad():
        z_audio = model.audio_forward(audio)
    return z_audio.mean(0).detach().cpu().float()


def get_ttmr_text_embedding(text: str, gpu: int = 0) -> torch.Tensor:
    model = ensure_model_loaded(gpu)
    with torch.no_grad():
        z_text = model.text_forward([text])
    return z_text.squeeze(0).detach().cpu().float()
