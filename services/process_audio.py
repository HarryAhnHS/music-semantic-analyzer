import numpy as np
import librosa
from services.clap_wrapper import CLAPWrapper
from services.llm_tagger import generate_tags_and_summary
def process_audio(file_path: str):
    # 1. Get CLAP embedding
    clap = CLAPWrapper()
    embedding = clap.get_embedding(file_path)

    # 2. Extract metadata
    y, sr = librosa.load(file_path, sr=16000)
    duration = float(librosa.get_duration(y=y, sr=sr))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)

    metadata = {
        "duration_sec": round(duration, 2),
        "tempo_bpm": round(float(tempo), 2),
        "chroma_vector": [round(float(c), 4) for c in chroma]
    }

    # Dummy LLM outputs
    tags, summary = generate_tags_and_summary(metadata)

    return {
        "embedding": [float(x) for x in embedding],
        "metadata": metadata,
        "tags": tags,
        "summary": summary
    }