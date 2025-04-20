import os
import requests
from tqdm import tqdm

# download the pretrained Ensure large checkpoint files are ignoredcheckpoint from huggingface
def download_checkpoint():
    url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt"
    output_path = "checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt"

    os.makedirs("checkpoints", exist_ok=True)

    print("downloading checkpoint...")

    if not os.path.exists(output_path):
        print("Downloading checkpoint...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    if chunk:
                        f.write(chunk)
    else:
        print("Checkpoint already exists.")

if __name__ == "__main__":
    download_checkpoint()