from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TAGGING_AUDIO_DIR = BASE_DIR / "data/tagging_index/fma/audio"
TRACKS_PATH = BASE_DIR / "data/tagging_index/fma/csv/tracks.csv"
GENRE_MAP_PATH = BASE_DIR / "data/tagging_index/fma/csv/genres.csv"

TAGGING_INDEX = BASE_DIR / "data/tagging_index/embeddings/clap_index.faiss"
TAGGING_META = BASE_DIR / "data/tagging_index/metadata/clap_metadata.json"

TTMR_INDEX = BASE_DIR / "data/tagging_index/embeddings/ttmr_index.faiss"
TTMR_META = BASE_DIR / "data/tagging_index/metadata/ttmr_metadata.json"
TTMR_ARTIST_INDEX = BASE_DIR / "data/tagging_index/embeddings/ttmr_artist_index.faiss"
TTMR_ARTIST_META = BASE_DIR / "data/tagging_index/metadata/ttmr_artist_metadata.json"

INTERNAL_INDEX = BASE_DIR / "data/matching_index/embeddings/internal_index.faiss"
INTERNAL_META = BASE_DIR / "data/matching_index/metadata/internal_metadata.json"
INTERNAL_TEXT_INDEX = BASE_DIR / "data/matching_index/embeddings/internal_text_index.faiss"
INTERNAL_TEXT_META = BASE_DIR / "data/matching_index/metadata/internal_text_metadata.json"

UPLOAD_DIR = BASE_DIR / "uploads/full"
UPLOADS_PREVIEW_DIR = BASE_DIR / "uploads/previews"
SEPARATED_DIR = BASE_DIR / "uploads/stems"