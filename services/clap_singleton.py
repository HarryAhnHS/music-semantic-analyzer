import os
import torch
import laion_clap
from functools import lru_cache

CKPT_PATH = "checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt"

# 🔐 Safety check: auto-download if missing
if not os.path.exists(CKPT_PATH):
    try:
        from scripts.download_clap_checkpoint import download_checkpoint
        print("[CLAP Model] Checkpoint missing, downloading...")
        download_checkpoint()
    except Exception as e:
        print(f"[CLAP Model] Failed to download checkpoint: {e}")

@lru_cache(maxsize=1)
def get_clap_model():
    """Lazy load CLAP model only when first needed"""
    print("[CLAP Model] Loading model on demand...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(CKPT_PATH)
    print("[CLAP Model] Model loaded successfully.")
    return model, device

# Lazy loading - models only loaded when first accessed
def get_clap_device():
    _, device = get_clap_model()
    return device

def get_clap_model_instance():
    model, _ = get_clap_model()
    return model