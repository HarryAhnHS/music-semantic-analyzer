import numpy as np
from pathlib import Path
import librosa
from demucs.separate import main as demucs_main
import os
from configs.index_configs import SEPARATED_DIR

def separate_stems(audio_path: str, model: str = "htdemucs", cache_dir: str = SEPARATED_DIR) -> dict:
    """
    Separates the given audio file into stems using Demucs and caches results.
    If stems already exist, it skips reprocessing.

    Returns a dict with paths to the stem files.
    """
    audio_path = Path(audio_path)
    track_name = audio_path.stem
    stem_dir = Path(cache_dir) / model / track_name

    # If stems already exist, skip separation
    if all((stem_dir / f"{s}.wav").exists() for s in ["vocals", "drums", "bass", "other"]):
        print(f"[Demucs] Stems already exist for: {track_name}")
    else:
        print(f"[Demucs] Separating stems for: {track_name}")
        os.makedirs(stem_dir.parent, exist_ok=True)
        # use segment 8 to reduce ram usage
        demucs_main([
            "--out", str(Path(cache_dir)),
            "-n", model,
            "--segment", "7",
            str(audio_path)
        ])

    return {
        "vocals": str(stem_dir / "vocals.wav"),
        "drums": str(stem_dir / "drums.wav"),
        "bass": str(stem_dir / "bass.wav"),
        "other": str(stem_dir / "other.wav")
    }


def classify_track_type(stems: dict) -> str:
    energy = {stem: np.mean(librosa.feature.rms(y=librosa.load(path, sr=None)[0])) for stem, path in stems.items()}
    vocal_energy = energy.get("vocals", 0)
    instrumental_energy = sum(energy.get(s, 0) for s in ["drums", "bass", "other"])
    
    if vocal_energy > instrumental_energy * 2:
        return "acapella"
    elif instrumental_energy > vocal_energy * 2:
        return "instrumental"
    return "song"
