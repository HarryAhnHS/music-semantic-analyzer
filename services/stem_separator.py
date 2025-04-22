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


import librosa
import numpy as np

def compute_rms_energy(path: str, sr: int = 22050) -> float:
    try:
        y, _ = librosa.load(path, sr=sr)
        if y is None or len(y) == 0:
            return 0.0
        rms = librosa.feature.rms(y=y)
        return np.mean(rms)
    except Exception as e:
        print(f"⚠️ Failed to compute RMS for {path}: {e}")
        return 0.0

def classify_track_type(stems: dict) -> str:
    energy = {
        stem: compute_rms_energy(path)
        for stem, path in stems.items()
    }

    print("🔍 Stem energy breakdown:", energy)

    # Filter out negligible stems (i.e., silence or bleed)
    min_energy_threshold = 1e-4
    filtered_energy = {k: v if v > min_energy_threshold else 0 for k, v in energy.items()}

    total_energy = sum(filtered_energy.values())
    if total_energy == 0:
        return "unknown"

    vocal_energy = filtered_energy.get("vocals", 0)
    instrumental_energy = sum(
        filtered_energy.get(s, 0) for s in ["drums", "bass", "other"]
    )

    vocal_ratio = vocal_energy / total_energy if total_energy else 0
    instrumental_ratio = instrumental_energy / total_energy if total_energy else 0

    print(f"🎧 Vocal Ratio: {vocal_ratio:.3f}, Instrumental Ratio: {instrumental_ratio:.3f}")

    if vocal_ratio > 0.3 and instrumental_ratio > 0.3:
        track_type = "song"
    elif vocal_ratio > 0.6:
        track_type = "acapella"
    elif instrumental_ratio > 0.6:
        track_type = "instrumental"
    else:
        track_type = "unknown"

    return {
        "energy": energy,
        "vocal_ratio": vocal_ratio,
        "instrumental_ratio": instrumental_ratio,
        "track_type": track_type
    }