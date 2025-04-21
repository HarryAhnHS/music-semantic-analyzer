import os
import json
from together import Together
from dotenv import load_dotenv

load_dotenv()
client = Together()

def generate_tags_and_summary(metadata: dict, neighbors: list[dict]) -> tuple[list[str], str]:
    chroma = metadata.get("chroma_vector", [])
    stem_chroma = metadata.get("stem_chroma_vector", [])
    tempo = metadata.get("tempo_bpm")

    title = metadata.get("title", "Unknown Title")
    artist = metadata.get("artist", "Unknown Artist")
    genre_top = metadata.get("genre", "")
    genre_names = metadata.get("genre_names", [])
    raw_tags = metadata.get("tags", [])
    location = metadata.get("location", "")
    track_type = metadata.get("track_type", "")
    stem_type = metadata.get("stem_type", "")

    # Enriched context
    artist_bio = metadata.get("artist_bio", "")
    album_description = metadata.get("album_description", "")

    # Format neighbors
    neighbor_summaries = []
    for n in neighbors[:5]:
        neighbor_summaries.append(
            f"""- "{n.get('title', 'Unknown')}" by {n.get('artist', 'Unknown')} ({n.get('genre', 'unknown genre')}) — tags: {", ".join(n.get('tags', [])) or 'none'}, location: {n.get('location', 'unknown')}, album: {n.get('album', '')}"""
        )
    neighbor_text = "\n".join(neighbor_summaries)

    # Dynamic instruction per stem
    extra_instructions = ""
    if stem_type == "vocals":
        extra_instructions = (
            "Focus on vocal qualities. Is the voice raspy, airy, autotuned, robotic, soft, deep, or nasal? "
            "Is it expressive, melodic, shouted, whispered? Describe vocal character, gender, and delivery style."
            "Chrome vector for vocals: {stem_chroma}"
        )
    elif stem_type == "drums":
        extra_instructions = (
            "Focus on the percussion style. Are there hi-hat rolls, trap triplets, hard kicks, rimshots, breakbeats? "
            "Mention groove, bounce, genre influences."
            "Chrome vector for drums: {stem_chroma}"
        )
    elif stem_type == "bass":
        extra_instructions = (
            "Describe the bassline's character. Is it 808-driven, sub-heavy, funky, jazz-influenced, synthy, or plucky?"
            "Chrome vector for bass: {stem_chroma}"
        )
    elif stem_type == "other":
        extra_instructions = (
            "Identify melodic instruments like guitar, synths, strings, piano, pads, or experimental sounds. "
            "Mention texture and atmosphere."
            "Chrome vector for melody: {stem_chroma}"
        )
    else:  # full track
        extra_instructions = (
            "If the track resembles a known artist or producer, mention it. "
            "Comment on feel, genre crossover, beat style, and if it's sample-heavy or electronic."
        )

    system_msg = (
        "You're an expert music producer and musicologist. Based on audio features, metadata, and similar songs, "
        "generate insightful tags and a natural-language summary of the track or stem's feel, style, instrumentation, and influences."
    )

    user_msg = f"""Input track details:
    - Title: {title}
    - Artist: {artist}
    - Location: {location}
    - Tempo: {tempo} BPM
    - Full Song Chroma Vector: {chroma}
    - Genre: {genre_top}
    - Genre Names: {", ".join(genre_names)}
    - Tags: {", ".join(raw_tags)}
    - Artist Bio: {artist_bio}
    - Album Description: {album_description}
    {f"- Stem Type: {stem_type}" if stem_type else f"- Track Type: {track_type}"}

    Here are a few similar tracks:
    {neighbor_text}

    {extra_instructions}

    ONLY RETURN RAW JSON (no markdown, no code blocks) with:
    1. "tags": a list of lowercase, descriptive semantic tags (e.g. "808s", "lil tecca type beat", "autotuned", "sample-heavy", "ambient synth", "drill drums")
    2. "summary": a 1-3 sentence paragraph describing the track/stem's vibe, style, and instrumentation.
    """

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
        return result["tags"], result["summary"]
    except Exception as e:
        print("Failed to parse LLM output:", e)
        print("Raw content was:\n", content)
        return ["unknown"], "Unable to generate summary."
