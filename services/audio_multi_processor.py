from services.llm_tagger import generate_tags_and_summary_hybrid
from services.metadata_extractor import extract_metadata
from configs.index_configs import (
    TAGGING_INDEX, TAGGING_META, 
    INTERNAL_INDEX, INTERNAL_META, 
    INTERNAL_TEXT_INDEX, INTERNAL_TEXT_META,
    TTMR_INDEX, TTMR_META, TTMR_ARTIST_INDEX, TTMR_ARTIST_META
)
from services.stem_separator import separate_stems, classify_track_type
from services.clap_manager import get_clap
from services.text_embedder import TextEmbeddingIndex
from services.ttmrpp_manager import get_ttmr
from utils.audio_utils import is_ignorable_stem
import json

def process_audio_hybrid(preview_path: str, full_path: str):
    # 1. Separate stems
    stems = separate_stems(preview_path)

    # 2. Classify track type
    track_info = classify_track_type(stems)

    # 3. Get CLAP embedding
    tagging_clap = get_clap(TAGGING_INDEX, TAGGING_META, read_only=True)
    clap_embedding = tagging_clap.get_embedding(preview_path)

    # 4. Get TTMR embedding and neighbors
    ttmr_embedder = get_ttmr(TTMR_INDEX, TTMR_META)
    ttmr_embedding = ttmr_embedder.get_audio_embedding(preview_path)
    overall_ttmr_neighbors = ttmr_embedder.query_neighbors_with_metadata(ttmr_embedding, k=3)
    # 4. Get TTMR artist embedding and neighbors
    ttmr_artist_embedder = get_ttmr(TTMR_ARTIST_INDEX, TTMR_ARTIST_META)
    overall_ttmr_artist_neighbors = ttmr_artist_embedder.query_neighbors_with_metadata(ttmr_embedding, k=3)

    # 5. Extract metadata
    overall_metadata = extract_metadata(full_path)
    overall_metadata["track_info"] = track_info
    

    # 6. Combine neighbors
    overall_clap_neighbors = tagging_clap.query_neighbors_with_tagging_metadata(clap_embedding, k=3)
    overall_hybrid_neighbors = overall_clap_neighbors + overall_ttmr_neighbors


    # 7. Generate tags and summary
    tags, summary = generate_tags_and_summary_hybrid(overall_metadata, overall_hybrid_neighbors, overall_ttmr_artist_neighbors)

    # 8. Stem-level tagging
    stem_tags = {}
    stem_summaries = {}

    for stem_name, stem_path in stems.items():
        stem_metadata = extract_metadata(stem_path)
        if is_ignorable_stem(stem_path):
            print(f"Ignoring stem: {stem_name}")
            stem_tags[stem_name] = []
            stem_summaries[stem_name] = ["Empty stem. The {stem_name} is silent and does not contain any audio content."]
            continue
        stem_embedding = tagging_clap.get_embedding(stem_path)
        clap_neighbors = tagging_clap.query_neighbors_with_tagging_metadata(stem_embedding, k=3)
        ttmr_embedding = ttmr_embedder.get_audio_embedding(stem_path)
        ttmr_neighbors = ttmr_embedder.query_neighbors_with_metadata(ttmr_embedding, k=3)
        ttmr_artist_neighbors = ttmr_artist_embedder.query_neighbors_with_metadata(ttmr_embedding, k=3)
        stem_meta = {
            **overall_metadata,
            "stem_chroma_vector": stem_metadata.get("chroma_vector", []),
            "stem_type": stem_name
        }
        hybrid_neighbors = clap_neighbors + ttmr_neighbors

        t, s = generate_tags_and_summary_hybrid(stem_meta, hybrid_neighbors, ttmr_artist_neighbors)
        stem_tags[stem_name] = t
        stem_summaries[stem_name] = s

    # 9. Add to CLAP index
    internal_clap = get_clap(INTERNAL_INDEX, INTERNAL_META, read_only=False)
    internal_metadata_entry = {
        "metadata": overall_metadata,
        "clap_neighbors": overall_clap_neighbors,
        "ttmr_neighbors": overall_ttmr_neighbors,
        "tags": tags,
        "summary": summary,
        "stem_tags": stem_tags,
        "stem_summaries": stem_summaries,
        "similar_artists": overall_ttmr_artist_neighbors
    }

    internal_clap.add_embedding_to_index(clap_embedding, internal_metadata_entry)
    internal_clap.save_index()

    # 10. Add to text search index
    text_index = TextEmbeddingIndex(INTERNAL_TEXT_INDEX, INTERNAL_TEXT_META)
    text_blob = text_index.generate_text_blob(internal_metadata_entry)
    internal_metadata_entry["text_embedding"] = text_blob
    print("text_blob", text_blob)
    text_index.add_entry(text_blob, internal_metadata_entry)
    text_index.save()

    return internal_metadata_entry
