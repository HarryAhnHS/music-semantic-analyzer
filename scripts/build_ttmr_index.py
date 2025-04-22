import os
import re
import json
import faiss
import numpy as np
from glob import glob
from time import time
from tqdm import tqdm
from datasets import load_dataset

from services.ttmrpp_manager import get_ttmr
from configs.index_configs import TAGGING_AUDIO_DIR, TTMR_INDEX, TTMR_META

from transformers import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

# ----------- Config -----------
BATCH_SIZE = 300
embedding_buffer = []
metadata_buffer = []
written, skipped_existing, skipped_missing, crashed = 0, 0, 0, 0
index = None

# ----------- Setup -----------
os.makedirs(os.path.dirname(TTMR_META), exist_ok=True)

print("🖎️ Loading enrich-fma-large dataset...")
dataset = load_dataset("seungheondoh/enrich-fma-large", split="train")
dataset_by_id = {str(entry["track_id"]): entry for entry in dataset}

if os.path.exists(TTMR_META):
    with open(TTMR_META, "r") as f:
        metadata_entries = json.load(f)
    existing_ids = set(str(entry["track_id"]) for entry in metadata_entries)
    print(f"⏭️ Resuming from {len(existing_ids)} previously processed entries.")
else:
    existing_ids = set()
    metadata_entries = []

if os.path.exists(TTMR_INDEX):
    print("📦 Loading existing FAISS index...")
    index = faiss.read_index(str(TTMR_INDEX))
else:
    print("🆕 Starting new FAISS index...")

# ----------- Discover Local MP3s -----------
all_mp3s = glob(os.path.join(TAGGING_AUDIO_DIR, "**/*.mp3"), recursive=True)
print(f"\n📁 Scanning directory: {TAGGING_AUDIO_DIR}")
print(f"🎵 MP3s found: {len(all_mp3s)}")
print(f"⏭️ Already processed: {len(existing_ids)}\n")

start_time = time()

# ---------- Load TTMR++ Singleton ----------
ttmr = get_ttmr(TTMR_INDEX, TTMR_META)

# ----------- Main Loop -----------
for path in tqdm(all_mp3s, desc="🎷 Embedding tracks"):
    filename = os.path.basename(path)
    match = re.match(r"(\d+)\.mp3", filename)
    if not match:
        skipped_missing += 1
        continue

    track_id = match.group(1)

    if track_id in existing_ids:
        skipped_existing += 1
        continue

    entry = dataset_by_id.get(track_id)
    if not entry:
        skipped_missing += 1
        continue

    try:
        emb = ttmr.get_audio_embedding(path).numpy().astype("float32")

        if index is None:
            index = faiss.IndexFlatL2(emb.shape[0])

        raw_tags = entry.get("tag_list", [])
        cleaned_tags = sorted(set(tag.lower().strip() for tag in raw_tags if isinstance(tag, str)))

        metadata = {
            "track_id": track_id,
            "title": entry.get("title", "").strip(),
            "artist": entry.get("artist_name", "").strip(),
            "caption": entry.get("pseudo_caption", "").strip(),
            "tags": cleaned_tags
        }

        embedding_buffer.append(emb)
        metadata_buffer.append(metadata)
        written += 1

        if written <= 5 or written % 25 == 0:
            print(f"📍 Embedded {track_id} ({written})")

        # Flush every BATCH_SIZE
        if len(embedding_buffer) >= BATCH_SIZE:
            print("💾 Flushing FAISS index and metadata to disk...")
            index.add(np.stack(embedding_buffer))
            metadata_entries.extend(metadata_buffer)

            faiss.write_index(index, str(TTMR_INDEX))
            with open(str(TTMR_META), "w") as f:
                json.dump(metadata_entries, f, indent=2)

            # Now safely update existing_ids
            existing_ids.update(str(m["track_id"]) for m in metadata_buffer)
            embedding_buffer.clear()
            metadata_buffer.clear()

    except Exception as e:
        print(f"[ERROR] Failed on {track_id}: {e}")
        crashed += 1
        continue

# ----------- Final Flush -----------
duration = time() - start_time

if embedding_buffer:
    print("🧬 Final batch flush...")
    if index is None:
        index = faiss.IndexFlatL2(embedding_buffer[0].shape[0])
    index.add(np.stack(embedding_buffer))
    metadata_entries.extend(metadata_buffer)

    faiss.write_index(index, str(TTMR_INDEX))
    with open(str(TTMR_META), "w") as f:
        json.dump(metadata_entries, f, indent=2)

    existing_ids.update(str(m["track_id"]) for m in metadata_buffer)

print("\n✅ All done!")
print(f"⏱️ Duration: {duration:.1f} sec")
print(f"✔️ Written: {written}")
print(f"⏭️ Skipped existing: {skipped_existing}")
print(f"🚫 Skipped missing/invalid: {skipped_missing}")
print(f"💥 Crashed: {crashed}")
