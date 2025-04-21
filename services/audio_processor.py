import numpy as np
from pathlib import Path
from services.clap_wrapper import CLAPWrapper
from services.llm_tagger import generate_tags_and_summary
from services.metadata_extractor import extract_metadata
from configs.index_configs import TAGGING_INDEX, TAGGING_META, INTERNAL_INDEX, INTERNAL_META
from services.stem_separator import separate_stems, classify_track_type

def process_audio(file_path: str):
    # 1. Separate stems
    stems = separate_stems(file_path)  # returns dict: { 'vocals': path, 'drums': path, ... }

    # 2. Heuristically classify track type
    track_type = classify_track_type(stems) # return string: "song", "instrumental", "vocal"

    # 3. Get main embedding (original audio)
    tagging_clap = CLAPWrapper(TAGGING_INDEX, TAGGING_META)
    embedding = tagging_clap.get_embedding(file_path)

    # 4. Extract audio metadata
    metadata = extract_metadata(file_path)

    # 5. Query tagging index for semantic neighbors (existing labeled tracks)
    top_neighbors = tagging_clap.query_neighbors_with_tagging_metadata(embedding, k=3)

    # 6. Generate tags + summary using LLM
    tags, summary = generate_tags_and_summary(metadata, top_neighbors)

    # 7. Add embedding to internal index - TODO:need to store metadata as well
    # internal_clap = CLAPWrapper(INTERNAL_INDEX, INTERNAL_META)
    # internal_clap.add_embedding_to_index(embedding)
    # internal_clap.save_index()

    return {
        "embedding": [float(x) for x in embedding],
        "track_type": track_type,
        "metadata": metadata,
        "neighbors": top_neighbors,
        "tags": tags,
        "summary": summary,
    }