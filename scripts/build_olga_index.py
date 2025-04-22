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

from services.ttmrpp_embedder import get_ttmr_audio_embedding
from configs.index_configs import TAGGING_AUDIO_DIR, TTMR_ARTIST_INDEX, TTMR_ARTIST_META, TTMR_META

BATCH_SIZE = 50

print("\U0001F4C2 Loading TTMR metadata...")
with open(TTMR_META, "r") as f:
    ttmr_meta = json.load(f)
track_to_artist = {entry["track_id"]: entry["artist"] for entry in ttmr_meta}
artist_to_track_ids = defaultdict(list)
for track_id, artist in track_to_artist.items():
    artist_to_track_ids[artist].append(track_id)

# Filter to artists with > 2 tracks
filtered_artists = {artist for artist, tracks in artist_to_track_ids.items() if len(tracks) >= 2}

print("\U0001F4C5 Loading OLGA dataset...")
olga = load_dataset("seungheondoh/olga-track-to-artist", split="train")

print("\U0001F9F9 Building fuzzy artist similarity map from OLGA...")
olga_artist_sim_map = defaultdict(Counter)
for entry in olga:
    sim_text = entry.get("sim_artist_text", "")
    names = [n.strip() for n in sim_text.split("[SEP]") if n.strip()]
    for name in names:
        olga_artist_sim_map[name.lower()].update(names)

existing_artist_names = set()
artist_metadata = []
embedding_buffer = []
metadata_buffer = []

if os.path.exists(TTMR_ARTIST_META):
    with open(TTMR_ARTIST_META, "r") as f:
        artist_metadata = json.load(f)
    existing_artist_names = set(entry["artist_name"] for entry in artist_metadata)

index = None
if os.path.exists(TTMR_ARTIST_INDEX):
    index = faiss.read_index(TTMR_ARTIST_INDEX)

print("\U0001F50E Searching local MP3s...")
all_mp3s = glob(os.path.join(TAGGING_AUDIO_DIR, "**/*.mp3"), recursive=True)
print(f"\n\U0001F3B5 MP3s found: {len(all_mp3s)}")

artist_to_embeddings = defaultdict(list)
artist_to_tracks = defaultdict(list)

for path in tqdm(all_mp3s, desc="\U0001F3BF Embedding"):
    filename = os.path.basename(path)
    match = re.match(r"(\d+)\.mp3", filename)
    if not match:
        continue
    track_id = match.group(1)
    artist_name = track_to_artist.get(track_id)
    if not artist_name or artist_name not in filtered_artists:
        continue
    try:
        emb = get_ttmr_audio_embedding(path).numpy().astype("float32")
        artist_to_embeddings[artist_name].append(emb)
        artist_to_tracks[artist_name].append(track_id)
    except Exception as e:
        print(f"[ERROR] Failed to embed {track_id}: {e}")

start_time = time()
written = 0

try:
    for artist, embs in artist_to_embeddings.items():
        if artist in existing_artist_names:
            continue
        if len(embs) < 2:
            continue

        avg_vector = np.mean(embs, axis=0).astype("float32")
        embedding_buffer.append(avg_vector)

        best_match, score = process.extractOne(artist.lower(), olga_artist_sim_map.keys())
        sim_name_counter = Counter()
        if score > 90:
            similar_artists = olga_artist_sim_map[best_match]
            for sim_name, count in similar_artists.items():
                if sim_name != artist.lower():
                    sim_name_counter[sim_name] += count

        top_sim_names = [name.title() for name, _ in sim_name_counter.most_common(5)]

        artist_metadata.append({
            "artist_name": artist,
            "track_ids": artist_to_tracks[artist],
            "sim_artist_names": top_sim_names
        })

        written += 1
        if written % BATCH_SIZE == 0:
            print("\U0001F4BE Flushing...")
            if index is None:
                index = faiss.IndexFlatL2(avg_vector.shape[0])
            index.add(np.stack(embedding_buffer))
            faiss.write_index(index, TTMR_ARTIST_INDEX)
            with open(TTMR_ARTIST_META, "w") as f:
                json.dump(artist_metadata, f, indent=2)
            embedding_buffer.clear()

except KeyboardInterrupt:
    print("\u274C Interrupted! Saving progress...")

finally:
    if embedding_buffer:
        print("\U0001F52C Final flush...")
        if index is None:
            index = faiss.IndexFlatL2(embedding_buffer[0].shape[0])
        index.add(np.stack(embedding_buffer))
        faiss.write_index(index, TTMR_ARTIST_INDEX)
        with open(TTMR_ARTIST_META, "w") as f:
            json.dump(artist_metadata, f, indent=2)

    print("\n\u2705 Done.")
    print(f"\U0000231B Duration: {time() - start_time:.1f}s")
    print(f"\U00002705 Artists written: {written}")
