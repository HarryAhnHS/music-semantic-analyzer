import numpy as np
import librosa
from services.clap_wrapper import CLAPWrapper
from services.llm_tagger import generate_tags_and_summary
from services.metadata_extractor import extract_metadata
from services.semantic_index import load_tag_index, query_tag_index  # new

# Load prebuilt semantic index (tag/summarization corpus)
# tag_index, tag_metadata = load_tag_index("data/tagging_index/embeddings/clap_index.faiss", "data/tagging_index/metadata/metadata.json")

# Load internal matching index (user uploads)
# internal_index, internal_metadata = load_internal_index(...)  # optional for matchmaking

def process_audio(file_path: str):
    # 1. Get CLAP embedding
    clap = CLAPWrapper()
    embedding = clap.get_embedding(file_path)

    # 2. Extract metadata (tempo, chroma, etc.)
    metadata = extract_metadata(file_path)

    # 3. Similarity search against pre-loaded external music semantic index (for tagging, summary)
    # top_neighbors = query_tag_index(embedding, tag_index, tag_metadata, k=3)

    # 4. Generate tags and summary using LLM with context
    # tags, summary = generate_tags_and_summary(metadata, top_neighbors)

    return {
        "embedding": [float(x) for x in embedding],
        "metadata": metadata,
        # "neighbors": top_neighbors,
        # "tags": tags,
        # "summary": summary
    }