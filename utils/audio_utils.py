import os
from pydub import AudioSegment
import librosa

# utils.py or inline in your script
def get_audio_path(audio_dir, track_id):
    """
    Given the base audio directory and a track_id, return full path to the corresponding MP3 file.

    Example:
        get_audio_path("./data/fma_small", 123) → "./data/fma_small/000/000123.mp3"
    """
    tid = int(track_id)
    return os.path.join(audio_dir, '{:03d}'.format(tid // 1000), '{:06d}.mp3'.format(tid))


# extract a 30 second preview from the center of the track
def extract_preview_segment(full_path: str, output_path: str, segment_duration_sec: int = 30):
    audio = AudioSegment.from_file(full_path)
    duration_ms = len(audio)
    segment_ms = segment_duration_sec * 1000

    # Center crop if possible, else take from start
    if duration_ms > segment_ms:
        start = (duration_ms - segment_ms) // 2
    else:
        start = 0

    preview = audio[start:start + segment_ms]
    preview.export(output_path, format="mp3")


import numpy as np


def is_stem_ignorable(y, sr, rms_thresh=0.01):
    if y is None or len(y) == 0:
        return True

    rms = librosa.feature.rms(y=y).mean()

    print(f"[Stem Check] RMS: {rms:.5f}")

    return rms < rms_thresh

import base64

def encode_audio_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")