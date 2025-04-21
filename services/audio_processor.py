from services.llm_tagger import generate_tags_and_summary
from services.metadata_extractor import extract_metadata
from configs.index_configs import TAGGING_INDEX, TAGGING_META, INTERNAL_INDEX, INTERNAL_META, INTERNAL_TEXT_INDEX, INTERNAL_TEXT_META
from services.stem_separator import separate_stems, classify_track_type
from services.clap_manager import get_clap
from services.clap_wrapper import CLAPWrapper
from services.text_embedder import TextEmbeddingIndex


def process_audio(preview_path: str, full_path: str):
    # 1. Separate stems
    stems = separate_stems(preview_path)  # returns dict: { 'vocals': path, 'drums': path, ... }

    # 2. Heuristically classify track type
    track_type = classify_track_type(stems) # return string: "song", "instrumental", "vocal"

    # 3. Get main embedding (original audio)
    tagging_clap = CLAPWrapper(TAGGING_INDEX, TAGGING_META, read_only=True)
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
        stem_metadata = extract_metadata(stem_path)
        stem_meta = {
            **metadata,
            "stem_chroma_vector": stem_metadata.get("chroma_vector", []),
            "stem_type": stem_name,
            "track_type": track_type
        }
        t, s = generate_tags_and_summary(stem_meta, neighbors)
        stem_tags[stem_name] = t
        stem_summaries[stem_name] = s

    # 7. Add audio embedding to internal index
    internal_clap = CLAPWrapper(INTERNAL_INDEX, INTERNAL_META, read_only=False)
    internal_metadata_entry = {
        "metadata": metadata,
        "neighbors": top_neighbors,
        "tags": tags,
        "summary": summary,
        "stem_tags": stem_tags,
        "stem_summaries": stem_summaries
    }
    print("internal_metadata_entry", internal_metadata_entry)
    internal_clap.add_embedding_to_index(embedding, internal_metadata_entry)
    internal_clap.save_index()

    # 8. Generate text embedding and encode to internal text index
    text_index = TextEmbeddingIndex(INTERNAL_TEXT_INDEX, INTERNAL_TEXT_META)
    text_blob = text_index.generate_text_blob(internal_metadata_entry)
    internal_metadata_entry["text_embedding"] = text_blob
    print("text_blob", text_blob)
    text_index.add_entry(text_blob, internal_metadata_entry)
    text_index.save()

    return internal_metadata_entry