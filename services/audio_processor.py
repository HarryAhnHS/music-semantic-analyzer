from services.llm_tagger import generate_tags_and_summary
from services.metadata_extractor import extract_metadata
from configs.index_configs import TAGGING_INDEX, TAGGING_META, INTERNAL_INDEX, INTERNAL_META
from services.stem_separator import separate_stems, classify_track_type
from services.clap_manager import get_clap

def process_audio(preview_path: str, full_path: str):
    # 1. Separate stems
    stems = separate_stems(preview_path)  # returns dict: { 'vocals': path, 'drums': path, ... }

    # 2. Heuristically classify track type
    track_type = classify_track_type(stems) # return string: "song", "instrumental", "vocal"

    # 3. Get main embedding (original audio)
    tagging_clap = get_clap(TAGGING_INDEX, TAGGING_META)
    embedding = tagging_clap.get_embedding(preview_path)

    # 4. Extract audio metadata
    metadata = extract_metadata(full_path)
    metadata["track_type"] = track_type

    # 5. Query tagging index for semantic neighbors (existing labeled tracks)
    top_neighbors = tagging_clap.query_neighbors_with_tagging_metadata(embedding, k=3)
    tags, summary = generate_tags_and_summary(metadata, top_neighbors)

    # 6. Stem-specific embeddings + tagging
    stem_tags = {}
    stem_summaries = {}

    for stem_name, stem_path in stems.items():
        stem_embedding = tagging_clap.get_embedding(stem_path)
        neighbors = tagging_clap.query_neighbors_with_tagging_metadata(stem_embedding, k=3)
        stem_meta = {
            **metadata,
            "stem_type": stem_name,
            "track_type": track_type
        }
        t, s = generate_tags_and_summary(stem_meta, neighbors)
        stem_tags[stem_name] = t
        stem_summaries[stem_name] = s

    # 7. Add embedding to internal index - TODO:need to store metadata as well
    # internal_clap = get_clap(INTERNAL_INDEX, INTERNAL_META)
    # internal_clap.add_embedding_to_index(embedding)
    # internal_clap.save_index()

    return {
        # "embedding": [float(x) for x in embedding],
        "metadata": metadata,
        # "neighbors": top_neighbors,
        "tags": tags,
        "summary": summary,
        "stem_tags": stem_tags,
        "stem_summaries": stem_summaries
    }