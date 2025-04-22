import os
import json
import faiss
from time import time
from tqdm import tqdm
from datasets import load_dataset
from services.ttmrpp_embedder import get_ttmr_audio_embedding
from configs.index_configs import TAGGING_AUDIO_DIR, TTMR_INDEX, TTMR_META
import numpy as np

import warnings
from transformers import logging

print("🖎️ Loading enrich-fma-large dataset...")
dataset = load_dataset("seungheondoh/enrich-fma-large", split="train")

print(dataset[0])
metadata = {
    "title": dataset[0].get("title", ""),
    "artist": dataset[0].get("artist", ""),
    "caption": dataset[0].get("caption", ""),
    "tags": dataset[0].get("tags", []),
    "track_id": dataset[0].get("track_id", "")
}

print(metadata)