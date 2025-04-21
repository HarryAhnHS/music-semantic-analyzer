import laion_clap
import torch

print("[CLAP Model] Loading global model...")
CLAP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP_MODEL = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
CLAP_MODEL.load_ckpt("checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt")
print("[CLAP Model] Model loaded successfully.")