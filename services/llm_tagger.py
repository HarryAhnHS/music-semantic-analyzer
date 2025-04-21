import os
import json
from together import Together
from dotenv import load_dotenv

load_dotenv()
client = Together()  # Uses TOGETHER_API_KEY from .env

def generate_tags_and_summary(metadata: dict, neighbors: list[dict]) -> tuple[list[str], str]:
    chroma = metadata.get("chroma_vector", [])
    tempo = metadata.get("tempo_bpm")
    duration = metadata.get("duration_sec")

    title = metadata.get("title", "Unknown Title")
    artist = metadata.get("artist", "Unknown Artist")
    genre_top = metadata.get("genre", "")
    genre_names = metadata.get("genre_names", [])
    raw_tags = metadata.get("tags", [])

    # Format neighbors for LLM context
    neighbor_summaries = []
    for n in neighbors:
        neighbor_summaries.append(f"""- "{n.get('title', 'Unknown')}" by {n.get('artist', 'Unknown')} — genre: {n.get('genre', '')}, tags: {", ".join(n.get("tags", []))}""")
    neighbor_text = "\n".join(neighbor_summaries)  # Limit to 5 neighbors

    system_msg = (
        "You're an expert music producer and musicologist. You receive audio-derived metadata and a few similar songs. "
        "Based on this, write a short natural description and assign descriptive tags for the given track."
    )

    user_msg = f"""Here is the audio metadata for a track:
    - Title: {title}
    - Artist: {artist}
    - Tempo: {tempo} BPM
    - Duration: {duration} seconds
    - Chroma Vector (pitch class intensity): {chroma}
    - Top Genre: {genre_top}
    - Genre Names: {", ".join(genre_names)}
    - Human Tags: {", ".join(raw_tags)}

    Here are similar songs to use as reference:
    {neighbor_text}

    ONLY RETURN RAW JSON (no markdown, no code blocks, no comments) with two fields:
    1. "tags": a list of 3–5 lowercase descriptive tags (e.g. "lo-fi", "vocals", "melancholic")
    2. "summary": a 1–2 sentence natural description of the track's feel, style, and instrumentation."""

    response = client.chat.completions.create(
        model=os.getenv("TOGETHER_MODEL"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
        print(result)
        return result["tags"], result["summary"]
    except Exception as e:
        print("Failed to parse LLM output:", e)
        print("Raw content was:\n", content)
        return ["unknown"], "Unable to generate summary."
