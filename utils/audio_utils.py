import os

# utils.py or inline in your script
def get_audio_path(audio_dir, track_id):
    """
    Given the base audio directory and a track_id, return full path to the corresponding MP3 file.

    Example:
        get_audio_path("./data/fma_small", 123) → "./data/fma_small/000/000123.mp3"
    """
    tid = int(track_id)
    return os.path.join(audio_dir, '{:03d}'.format(tid // 1000), '{:06d}.mp3'.format(tid))
