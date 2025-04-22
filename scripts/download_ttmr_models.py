import os
import requests
from tqdm import tqdm

def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"{os.path.basename(output_path)} already exists.")
        return
    print(f"Downloading {os.path.basename(output_path)}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)

def download_ttmrpp():
    download_file(
        "https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.pth",
        "models/ttmrpp/best.pth"
    )
    download_file(
        "https://huggingface.co/seungheondoh/ttmr-pp/resolve/main/ttmrpp_resnet_roberta.yaml",
        "models/ttmrpp/hparams.yaml"
    )

if __name__ == "__main__":
    download_ttmrpp()