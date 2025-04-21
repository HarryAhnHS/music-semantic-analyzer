import torch

MODEL_PATH = "checkpoints/instrument_model.pt"
LABELS = ["drums", "bass", "guitar", "piano", "synth", "strings", "vocals", "saxophone"]
SAMPLE_RATE = 22050
DURATION = 5.0  # seconds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"