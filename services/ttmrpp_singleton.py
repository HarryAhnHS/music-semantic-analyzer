# services/ttmr_singleton.py
from mtrpp.utils.eval_utils import load_ttmr_pp
import torch

print("[TTMR++] Loading shared model...")
TTMR_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TTMR_MODEL, _, _ = load_ttmr_pp("models/ttmrpp", model_types="best")
TTMR_MODEL = TTMR_MODEL.to(TTMR_DEVICE).eval()
print("[TTMR++] Model loaded.")