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
warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

# Prepare output directory
os.makedirs(os.path.dirname(TTMR_META), exist_ok=True)

print("🖎️ Loading enrich-fma-large dataset...")
dataset = load_dataset("seungheondoh/enrich-fma-large", split="train")

# Load previously saved metadata (if exists)
if os.path.exists(TTMR_META):
    with open(TTMR_META, "r") as f:
        existing_metadata = json.load(f)
    existing_ids = set(int(entry["track_id"]) for entry in existing_metadata)
    metadata_entries = existing_metadata
    print(f"⏭️ Loaded {len(existing_ids)} previously processed entries.")
else:
    existing_ids = set()
    metadata_entries = []

total = len(dataset)
written, skipped_existing, skipped_missing, crashed = 0, 0, 0, 0
embeddings = []

start_time = time()

print(f"\n📆 Audio directory: {TAGGING_AUDIO_DIR}")
print(f"🌟 Total tracks in metadata: {total}")
print(f"⏭️ Skipping {len(existing_ids)} already embedded\n")

try:
    for i, entry in enumerate(tqdm(dataset, desc="🎷 Embedding tracks")):
        track_id = entry["track_id"]

        if int(track_id) in existing_ids:
            print(f"⏭️ Skipping already processed track ID {track_id}")
            skipped_existing += 1
            continue

        folder = track_id.zfill(6)[:3]
        filename = f"{track_id.zfill(6)}.mp3"
        abs_path = os.path.join(TAGGING_AUDIO_DIR, folder, filename)

        if not os.path.exists(abs_path):
            skipped_missing += 1
            continue

        try:
            embedding = get_ttmr_audio_embedding(abs_path)

            # ✅ Extract and clean metadata
            raw_tags = entry.get("tag_list", [])
            cleaned_tags = sorted(set(tag.lower().strip() for tag in raw_tags if isinstance(tag, str)))

            metadata = {
                "track_id": track_id,
                "title": entry.get("title", "").strip(),
                "artist": entry.get("artist_name", "").strip(),
                "caption": entry.get("pseudo_caption", "").strip(),
                "tags": cleaned_tags
            }

            embeddings.append(embedding.numpy().astype("float32"))
            metadata_entries.append(metadata)

            written += 1
            if written <= 5 or written % 25 == 0:
                print(f"📍 [{written}] Confirmed locally: track_id={track_id}")
                print(f"✅ Added embedding + metadata for {len(embeddings)} embeddings")

            if written < 1:
                print(metadata_entries[0])

        except Exception as e:
            print(f"[ERROR] {track_id} failed: {e}")
            crashed += 1
            continue

finally:
    duration = time() - start_time

    if embeddings:
        print("\n🧬 Building FAISS index...")
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.stack(embeddings))

        print(f"📆 Saving FAISS index to {TTMR_INDEX}")
        faiss.write_index(index, str(TTMR_INDEX))

        print(f"📝 Writing metadata to {TTMR_META}")
        with open(str(TTMR_META), "w") as f:
            json.dump(metadata_entries, f, indent=2)

    print("\n✅ Done embedding and indexing.")
    print(f"⏱️ Duration: {duration:.1f} seconds")
    print(f"✔️ Written: {written}")
    print(f"⏭️ Skipped existing: {skipped_existing}")
    print(f"🚫 Skipped missing files: {skipped_missing}")
    print(f"💥 Crashed: {crashed}")
