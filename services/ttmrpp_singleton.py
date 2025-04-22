# services/ttmr_singleton.py
from mtrpp.utils.eval_utils import load_ttmr_pp
import torch
import os

# 🔐 Optional fallback download logic
if not os.path.exists("models/ttmrpp/best.pth"):
    try:
        from scripts.download_ttmr_models import download_ttmrpp
        print("[TTMR++] Checkpoint missing, downloading now...")
        download_ttmrpp()
    except Exception as e:
        print(f"[TTMR++] Failed to auto-download checkpoint: {e}")

print("[TTMR++] Loading shared model...")
TTMR_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TTMR_MODEL, _, _ = load_ttmr_pp("models/ttmrpp", model_types="best")
TTMR_MODEL = TTMR_MODEL.to(TTMR_DEVICE).eval()
print("[TTMR++] Model loaded.")