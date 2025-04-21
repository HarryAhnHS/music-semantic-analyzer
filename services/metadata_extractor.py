import librosa

def extract_metadata(file_path: str):
    y, sr = librosa.load(file_path, sr=16000)
    duration = float(librosa.get_duration(y=y, sr=sr))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)

    metadata = {
        "duration_sec": round(duration, 2),
        "tempo_bpm": round(float(tempo), 2),
        "chroma_vector": [round(float(c), 4) for c in chroma]
    }

    return metadata