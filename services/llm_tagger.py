import os
import json
from together import Together
from dotenv import load_dotenv

load_dotenv()
client = Together()

def generate_tags_and_summary(metadata: dict, neighbors: list[dict]) -> tuple[list[str], str]:
    chroma = metadata.get("chroma_vector", [])
    tempo = metadata.get("tempo_bpm")
    duration = metadata.get("duration_sec")

    title = metadata.get("title", "Unknown Title")
    artist = metadata.get("artist", "Unknown Artist")
    genre_top = metadata.get("genre", "")
    genre_names = metadata.get("genre_names", [])
    raw_tags = metadata.get("tags", [])

    # Extended metadata
    artist_bio = metadata.get("artist_bio", "")
    album_description = metadata.get("album_description", "")
    location = metadata.get("location", "")

    # Format neighbors
    neighbor_summaries = []
    for n in neighbors[:5]:  # Limit to top 5
        neighbor_summaries.append(
            f"""- "{n.get('title', 'Unknown')}" by {n.get('artist', 'Unknown')} ({n.get('genre', 'unknown genre')}) — tags: {", ".join(n.get('tags', [])) or 'none'}, location: {n.get('location', 'unknown')}, album: {n.get('album', '')}"""
        )
    neighbor_text = "\n".join(neighbor_summaries)

    system_msg = (
        "You're an expert music producer and musicologist. You're given audio-derived features "
        "and rich human metadata about a track and a few similar songs. Use this to write a short natural-language summary "
        "of the input song's feel, style, and instrumentation, and provide descriptive tags."
    )

    user_msg = f"""Input track details:
    - Title: {title}
    - Artist: {artist}
    - Location: {location}
    - Tempo: {tempo} BPM
    - Duration: {duration} seconds
    - Chroma Vector (pitch class intensity): {chroma}
    - Genre: {genre_top}
    - Genre Names: {", ".join(genre_names)}
    - Tags: {", ".join(raw_tags)}
    - Artist Bio: {artist_bio}
    - Album Description: {album_description}

    Here are a few similar tracks:
    {neighbor_text}

    ONLY RETURN RAW JSON (no markdown, no code blocks, no comments) with two fields:
    1. "tags": a list of lowercase descriptive tags (e.g. "lo-fi", "vocals", "melancholic", "opium")
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
