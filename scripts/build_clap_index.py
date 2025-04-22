import os
import json
from tqdm import tqdm
import pandas as pd
from time import time
from bs4 import BeautifulSoup

from utils.audio_utils import get_audio_path
from configs.index_configs import TAGGING_AUDIO_DIR, TRACKS_PATH, GENRE_MAP_PATH, TAGGING_INDEX, TAGGING_META
from services.clap_manager import get_clap

def clean_html(text, max_len=500):
    """Strip HTML and truncate long strings."""
    if not text or not isinstance(text, str):
        return ""
    clean = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    return clean[:max_len]

def safe_eval(val):
    """Safely eval list-like string fields."""
    try:
        return eval(val) if isinstance(val, str) else []
    except:
        return []

def load_genre_map(genre_csv_path):
    df = pd.read_csv(genre_csv_path)
    return {row["genre_id"]: row["title"] for _, row in df.iterrows()}

def main():
    print("🔄 Initializing CLAP model and loading metadata...")
    clap = get_clap(TAGGING_INDEX, TAGGING_META, read_only=False)

    metadata = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])
    genre_map = load_genre_map(GENRE_MAP_PATH)

    # Load previously processed metadata if available
    if TAGGING_META.exists():
        with open(TAGGING_META, "r") as f:
            output_meta = json.load(f)
        existing_ids = set(entry["id"] for entry in output_meta)
        print(f"⚠️ Found existing metadata: skipping {len(existing_ids)} already processed tracks.")
    else:
        output_meta = []
        existing_ids = set()

    # Subset to only tracks we need to process
    track_ids_to_process = [tid for tid in metadata.index if int(tid) not in existing_ids]
    total = len(track_ids_to_process)

    print(f"\n📦 Dataset: {TAGGING_AUDIO_DIR.name}")
    print(f"🎯 Attempting to process {total} new tracks...\n")

    processed = 0
    skipped_missing = 0
    skipped_existing = 0
    skipped_invalid = 0

    start_time = time()

    try:
        for i, track_id in enumerate(tqdm(track_ids_to_process, desc="🎧 Processing tracks")):
            path = get_audio_path(TAGGING_AUDIO_DIR, track_id)
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
                    "title": str(row.get(("track", "title"), "")),
                    "artist": str(row.get(("artist", "name"), "")),
                    "album": str(row.get(("album", "title"), "")),
                    "genre": str(row.get(("track", "genre_top")) or "").lower(),
                    "genre_names": [str(g).lower() for g in genre_names if g],
                    "duration": float(row.get(("track", "duration"))) if pd.notnull(row.get(("track", "duration"))) else None,
                    "tags": [str(t).lower() for t in safe_eval(row.get(("track", "tags")))],
                    "artist_bio": clean_html(row.get(("artist", "bio"), ""), max_len=400),
                    "artist_projects": clean_html(row.get(("artist", "related_projects"), ""), max_len=300),
                    "artist_website": str(row.get(("artist", "website"), "")),
                    "album_description": clean_html(row.get(("album", "information"), ""), max_len=400),
                    "album_engineer": str(row.get(("album", "engineer"), "")),
                    "license": str(row.get(("track", "license"), "")),
                    "location": str(row.get(("artist", "location"), "")),
            })

            processed += 1
            if processed <= 10 or processed % 25 == 0:
                print(f"✅ [{processed}] Processed track {track_id} ({i+1}/{total})")

    finally:
        duration = time() - start_time
        print("\n💾 Saving FAISS index and metadata...")
        os.makedirs(os.path.dirname(TAGGING_META), exist_ok=True)
        clap.save_index()
        with open(TAGGING_META, "w") as f:
            json.dump(output_meta, f, indent=2)

        print(f"\n🎉 Done in {duration:.1f}s")
        print(f"✔️  New tracks processed: {processed}")
        print(f"↪️  Skipped existing (preloaded): {len(existing_ids)}")
        print(f"🚫 Skipped missing files: {skipped_missing}")
        print(f"⚠️  Skipped invalid embeddings: {skipped_invalid}")

if __name__ == "__main__":
    main()
