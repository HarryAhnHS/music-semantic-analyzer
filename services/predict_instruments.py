# import torch
# import librosa
# import numpy as np
# import torch.nn.functional as F
# from pathlib import Path
# from torchvision import transforms
# from model import AudioClassifier  # import model from their repo
# import torchaudio

# # --------- Config ---------
# MODEL_PATH = "checkpoints/instrument_model.pt"
# LABELS = ["drums", "bass", "guitar", "piano", "synth", "strings", "vocals", "saxophone"]
# SAMPLE_RATE = 22050
# DURATION = 5.0  # seconds
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --------- Load model ---------
# model = AudioClassifier(num_classes=len(LABELS))
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.to(DEVICE)
# model.eval()


# # --------- Prediction ---------
# def predict_instruments_from_mp3(file_path: str):
#     # Load and resample
#     waveform, sr = torchaudio.load(file_path)
#     if sr != SAMPLE_RATE:
#         waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)

#     # Convert to mono
#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)

#     # Cut/pad to DURATION seconds
#     target_length = int(SAMPLE_RATE * DURATION)
#     if waveform.shape[1] < target_length:
#         pad = target_length - waveform.shape[1]
#         waveform = F.pad(waveform, (0, pad))
#     else:
#         waveform = waveform[:, :target_length]

#     # Convert to log-mel spectrogram
#     mel_spec = torchaudio.transforms.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_fft=1024,
#         hop_length=512,
#         n_mels=128
#     )(waveform)

#     log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
#     input_tensor = log_mel_spec.unsqueeze(0).to(DEVICE)  # shape: [1, 1, 128, time]

#     # Predict
#     with torch.no_grad():
#         logits = model(input_tensor)
#         probs = torch.sigmoid(logits).squeeze()

#     # Get top instruments
#     probs_np = probs.cpu().numpy()
#     result = [
#         {"instrument": LABELS[i], "confidence": float(probs_np[i])}
#         for i in np.argsort(-probs_np)[:5]
#         if probs_np[i] > 0.2  # threshold
#     ]
#     return result
