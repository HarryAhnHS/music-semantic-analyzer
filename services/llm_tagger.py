import os
import json
from together import Together
from dotenv import load_dotenv

load_dotenv()
client = Together()  # Uses TOGETHER_API_KEY from .env

def generate_tags_and_summary(metadata: dict) -> tuple[list[str], str]:
    chroma = metadata.get("chroma_vector", [])
    tempo = metadata.get("tempo_bpm")
    duration = metadata.get("duration_sec")

    system_msg = (
        "You're an expert music producer. You receive extracted features from an MP3 file "
        "like tempo, duration, and pitch distribution. Based on this, describe the song's style, mood, and instrumentation."
    )
    
    user_msg = f"""Here is the audio metadata:
    - Tempo: {tempo} BPM
    - Duration: {duration} seconds
    - Chroma Vector (pitch class intensity): {chroma}

    ONLY RETURN RAW JSON (no markdown, no code blocks, no comments) with two fields:
    1. "tags": a list of 3–5 lowercase descriptive tags (e.g. "lo-fi", "vocals", "melancholic")
    2. "summary": a 1–2 sentence natural description of the track."""

    response = client.chat.completions.create(
        model=os.getenv("TOGETHER_MODEL"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content

    # Try to parse response into JSON object
    try:
        result = json.loads(content)  # Quick and dirty, replace with json.loads if needed
        print(result)
        return result["tags"], result["summary"]
    except Exception as e:
        print("Failed to parse LLM output:", e)
        print("Raw content was:\n", content)
        return ["unknown"], "Unable to generate summary."
