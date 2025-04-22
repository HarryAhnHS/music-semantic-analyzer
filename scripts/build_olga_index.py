import os
import json
import re
import faiss
import numpy as np
from glob import glob
from collections import defaultdict, Counter
from tqdm import tqdm
from datasets import load_dataset
from fuzzywuzzy import process
from time import time
from transformers import logging
import warnings

from services.ttmrpp_manager import get_ttmr
from configs.index_configs import TAGGING_AUDIO_DIR, TTMR_ARTIST_INDEX, TTMR_ARTIST_META, TTMR_META

warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

# ---------- Config ----------
BATCH_SIZE = 50
print("\n🚀 Starting artist index build with resume + batch support...")

# ---------- Load Metadata ----------
print("📂 Loading TTMR metadata...")
with open(TTMR_META, "r") as f:
    ttmr_meta = json.load(f)

track_to_artist = {entry["track_id"]: entry["artist"] for entry in ttmr_meta}

# Filter for artists with 2+ tracks
artist_counts = Counter(track_to_artist.values())
valid_artists = {artist for artist, count in artist_counts.items() if count >= 2}

# ---------- Load OLGA ----------
print("📅 Loading OLGA dataset...")
olga = load_dataset("seungheondoh/olga-track-to-artist", split="train")
olga_artist_sim_map = defaultdict(Counter)

print("🧹 Building fuzzy artist similarity map...")
for entry in olga:
    sim_text = entry.get("sim_artist_text", "")
    names = [n.strip() for n in sim_text.split("[SEP]") if n.strip()]
    for name in names:
        olga_artist_sim_map[name.lower()].update(names)

# ---------- Resume Setup ----------
if os.path.exists(TTMR_ARTIST_META):
    with open(TTMR_ARTIST_META, "r") as f:
        artist_metadata = json.load(f)
    processed_artists = set(entry["artist_name"] for entry in artist_metadata)
    print(f"🔁 Resuming from {len(processed_artists)} artists.")
else:
    artist_metadata = []
    processed_artists = set()

if os.path.exists(TTMR_ARTIST_INDEX):
    print("📦 Loading existing FAISS index...")
    index = faiss.read_index(str(TTMR_ARTIST_INDEX))
else:
    index = None

# ---------- Collect Valid MP3 Paths ----------
print("🔎 Scanning MP3s...")
all_mp3s = glob(os.path.join(TAGGING_AUDIO_DIR, "**/*.mp3"), recursive=True)

artist_to_paths = defaultdict(list)
track_to_path = {}

for path in all_mp3s:
    match = re.match(r"(\d+)\.mp3", os.path.basename(path))
    if not match:
        continue
    track_id = match.group(1)
    artist = track_to_artist.get(track_id)
    if artist in valid_artists:
        artist_to_paths[artist].append(path)
        track_to_path[track_id] = path

# ---------- Load TTMR++ Singleton ----------
ttmr = get_ttmr(TTMR_ARTIST_INDEX, TTMR_ARTIST_META)

# ---------- Main Loop ----------
embedding_buffer = []
metadata_buffer = []
written, skipped, crashed = 0, 0, 0
start_time = time()

try:
    for artist, paths in tqdm(artist_to_paths.items(), desc="🎨 Artists"):
        if artist in processed_artists:
            skipped += 1
            continue

        try:
            embs = []
            track_ids = []

            for path in paths:
                track_id = re.match(r"(\d+)\.mp3", os.path.basename(path)).group(1)
                emb = ttmr.get_audio_embedding(path)
                embs.append(emb)
                track_ids.append(track_id)

            if len(embs) == 0:
                continue

            avg_vector = np.mean(embs, axis=0).astype("float32")
            embedding_buffer.append(avg_vector)

            # Fuzzy similarity via OLGA
            sim_name_counter = Counter()
            best_match, score = process.extractOne(artist.lower(), olga_artist_sim_map.keys())

            if score > 90:
                for sim_name, count in olga_artist_sim_map[best_match].items():
                    if sim_name != artist.lower():
                        sim_name_counter[sim_name] += count

            top_sim_names = [name.title() for name, _ in sim_name_counter.most_common(5)]

            artist_metadata_entry = {
                "artist_name": artist,
                "track_ids": track_ids,
                "sim_artist_names": top_sim_names
            }

            metadata_buffer.append(artist_metadata_entry)
            written += 1

            if written <= 3 or written % 10 == 0:
                print(f"📌 Added {artist} with {len(track_ids)} tracks")

            # Flush
            if len(metadata_buffer) >= BATCH_SIZE:
                print("💾 Writing batch to disk...")
                if index is None:
                    index = faiss.IndexFlatL2(avg_vector.shape[0])
                index.add(np.stack(embedding_buffer))

                artist_metadata.extend(metadata_buffer)
                faiss.write_index(index, TTMR_ARTIST_INDEX)
                with open(str(TTMR_ARTIST_META), "w") as f:
                    json.dump(artist_metadata, f, indent=2)

                processed_artists.update(entry["artist_name"] for entry in metadata_buffer)
                embedding_buffer.clear()
                metadata_buffer.clear()

        except Exception as e:
            print(f"[ERROR] Failed on {artist}: {e}")
            crashed += 1

finally:
    # ---------- Final Flush ----------
    if embedding_buffer:
        print("🧬 Final batch flush...")
        if index is None:
            index = faiss.IndexFlatL2(embedding_buffer[0].shape[0])
        index.add(np.stack(embedding_buffer))
        artist_metadata.extend(metadata_buffer)
        faiss.write_index(index, str(TTMR_ARTIST_INDEX))
        with open(str(TTMR_ARTIST_META), "w") as f:
            json.dump(artist_metadata, f, indent=2)

    end_time = time()

    print("\n✅ Done.")
    print(f"⏱️ Duration: {end_time - start_time:.1f}s")
    print(f"✔️ Embedded: {written} artists")
    print(f"⏭️ Skipped (already processed): {skipped}")
    print(f"💥 Crashed: {crashed}")