import torch
import librosa
import numpy as np
import laion_clap

print(torch.__version__)

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


class CLAPWrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel= 'HTSAT-base'
        )
        self.model.load_ckpt("checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt")

    def get_embedding(self, file_path: str) -> list[float]:
        # Use librosa to load and resample to 48kHz
        audio_data, _ = librosa.load(file_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)  # (1, T)
        
        # Optional: apply quantization (matches training input domain)
        audio_data = float32_to_int16(audio_data)
        audio_data = int16_to_float32(audio_data)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_data).float().to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.model.get_audio_embedding_from_data(audio_tensor, use_tensor=True)

        return embedding.squeeze().cpu().numpy().tolist()