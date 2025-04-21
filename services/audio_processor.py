import numpy as np
from pathlib import Path
from services.clap_wrapper import CLAPWrapper
from services.llm_tagger import generate_tags_and_summary
from services.metadata_extractor import extract_metadata
from configs.index_configs import TAGGING_INDEX, TAGGING_META, INTERNAL_INDEX, INTERNAL_META

def process_audio(file_path: str):
    # 1. Get CLAP embedding (shared)
    clap_query = CLAPWrapper(TAGGING_INDEX, TAGGING_META)
    embedding = clap_query.get_embedding(file_path)

    # 2. Extract audio metadata
    metadata = extract_metadata(file_path)

    # 3. Query tagging index for semantic neighbors (existing labeled tracks)
    top_neighbors = clap_query.query_neighbors_with_tagging_metadata(embedding, k=3)

    # 4. Generate tags + summary using LLM
    tags, summary = generate_tags_and_summary(metadata, top_neighbors)

    clap_store = CLAPWrapper(INTERNAL_INDEX, INTERNAL_META)
    clap_store.add_embedding_to_index(embedding)
    clap_store.save_index()

    return {
        "embedding": [float(x) for x in embedding],
        "metadata": metadata,
        "neighbors": top_neighbors,
        "tags": tags,
        "summary": summary
    }