import os
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from time import time

from services.clap_wrapper import CLAPWrapper
from utils.audio_utils import get_audio_path

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data/tagging_index/fma/audio"
METADATA_PATH = BASE_DIR / "data/tagging_index/fma/csv/tracks.csv"
GENRE_MAP_PATH = BASE_DIR / "data/tagging_index/fma/csv/genres.csv"
OUTPUT_INDEX = BASE_DIR / "data/tagging_index/embeddings/clap_index.faiss"
OUTPUT_META = BASE_DIR / "data/tagging_index/metadata/metadata.json"

def load_genre_map(genre_csv_path):
    df = pd.read_csv(genre_csv_path)
    return {row["genre_id"]: row["title"] for _, row in df.iterrows()}

def main():
    print("🔄 Initializing CLAP model and loading metadata...")
    clap = CLAPWrapper(faiss_path=OUTPUT_INDEX)

    metadata = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])
    genre_map = load_genre_map(GENRE_MAP_PATH)

    # Load previously processed metadata if available
    if OUTPUT_META.exists():
        with open(OUTPUT_META, "r") as f:
            output_meta = json.load(f)
        existing_ids = set(entry["id"] for entry in output_meta)
        print(f"⚠️ Found existing metadata: skipping {len(existing_ids)} already processed tracks.")
    else:
        output_meta = []
        existing_ids = set()

    # Subset to only tracks we need to process
    track_ids_to_process = [tid for tid in metadata.index if int(tid) not in existing_ids]
    total = len(track_ids_to_process)

    print(f"\n📦 Dataset: {AUDIO_DIR.name}")
    print(f"🎯 Attempting to process {total} new tracks...\n")

    processed = 0
    skipped_missing = 0
    skipped_existing = 0
    skipped_invalid = 0

    start_time = time()

    try:
        for i, track_id in enumerate(tqdm(track_ids_to_process, desc="🎧 Processing tracks")):
            path = get_audio_path(AUDIO_DIR, track_id)
            if not os.path.exists(path):
                skipped_missing += 1
                continue

            try:
                emb = clap.process_and_index(path)
                if not isinstance(emb, list) or len(emb) != 512:
                    skipped_invalid += 1
                    continue
            except Exception as e:
                print(f"[CLAP crash] {track_id} - {e}")
                continue

            # Process metadata
            row = metadata.loc[track_id]
            genre_ids = eval(row.get(("track", "genres_all"), "[]"))
            genre_names = [genre_map.get(gid) for gid in genre_ids if gid in genre_map]

            output_meta.append({
                "id": int(track_id),
                "title": row.get(("track", "title"), ""),
                "artist": row.get(("artist", "name"), ""),
                "genre": row.get(("track", "genre_top"), ""),
                "genre_names": genre_names,
                "tags": eval(row.get(("track", "tags"), "[]")) if isinstance(row.get(("track", "tags"), None), str) else [],
            })

            processed += 1
            if processed <= 10 or processed % 25 == 0:
                print(f"✅ [{processed}] Processed track {track_id} ({i+1}/{total})")

    finally:
        duration = time() - start_time
        print("\n💾 Saving FAISS index and metadata...")
        os.makedirs(os.path.dirname(OUTPUT_META), exist_ok=True)
        clap.save_index()
        with open(OUTPUT_META, "w") as f:
            json.dump(output_meta, f, indent=2)

        print(f"\n🎉 Done in {duration:.1f}s")
        print(f"✔️  New tracks processed: {processed}")
        print(f"↪️  Skipped existing (preloaded): {len(existing_ids)}")
        print(f"🚫 Skipped missing files: {skipped_missing}")
        print(f"⚠️  Skipped invalid embeddings: {skipped_invalid}")

if __name__ == "__main__":
    main()
