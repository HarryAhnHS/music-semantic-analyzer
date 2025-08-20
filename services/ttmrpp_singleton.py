# services/ttmr_singleton.py
import os
import torch
from functools import lru_cache
from external.music_text_representation_pp.mtrpp.utils.eval_utils import load_ttmr_pp

# 🔐 Optional fallback download logic
if not os.path.exists("models/ttmrpp/best.pth"):
    try:
        from scripts.download_ttmr_models import download_ttmrpp
        print("[TTMR++] Checkpoint missing, downloading now...")
        download_ttmrpp()
    except Exception as e:
        print(f"[TTMR++] Failed to auto-download checkpoint: {e}")

@lru_cache(maxsize=1)
def get_ttmr_model():
    """Lazy load TTMR++ model only when first needed"""
    print("[TTMR++] Loading shared model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_ttmr_pp("models/ttmrpp", model_types="best")
    model = model.to(device).eval()
    print("[TTMR++] Model loaded.")
    return model, device

# Lazy loading - models only loaded when first accessed
def get_ttmr_device():
    _, device = get_ttmr_model()
    return device

def get_ttmr_model_instance():
    model, _ = get_ttmr_model()
    return model