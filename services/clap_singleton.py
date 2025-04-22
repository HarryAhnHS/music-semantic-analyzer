import os
import torch
import laion_clap

CKPT_PATH = "checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt"

# 🔐 Safety check: auto-download if missing
if not os.path.exists(CKPT_PATH):
    try:
        from scripts.download_clap_checkpoint import download_checkpoint
        print("[CLAP Model] Checkpoint missing, downloading...")
        download_checkpoint()
    except Exception as e:
        print(f"[CLAP Model] Failed to download checkpoint: {e}")


print("[CLAP Model] Loading global model...")
CLAP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP_MODEL = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
CLAP_MODEL.load_ckpt("checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt")
print("[CLAP Model] Model loaded successfully.")