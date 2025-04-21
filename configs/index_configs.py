from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TAGGING_AUDIO_DIR = BASE_DIR / "data/tagging_index/fma/audio"
TRACKS_PATH = BASE_DIR / "data/tagging_index/fma/csv/tracks.csv"
GENRE_MAP_PATH = BASE_DIR / "data/tagging_index/fma/csv/genres.csv"

TAGGING_INDEX = BASE_DIR / "data/tagging_index/embeddings/clap_index.faiss"
TAGGING_META = BASE_DIR / "data/tagging_index/metadata/metadata.json"

INTERNAL_INDEX = BASE_DIR / "data/matching_index/embeddings/internal_index.faiss"
INTERNAL_META = BASE_DIR / "data/matching_index/metadata/internal_metadata.json"

UPLOAD_DIR = BASE_DIR / "uploads"

SEPARATED_DIR = BASE_DIR / "upload_stems"