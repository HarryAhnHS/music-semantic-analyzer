from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os, shutil, uuid, tempfile, time
from services.audio_multi_processor import process_audio_hybrid
from configs.index_configs import UPLOAD_DIR, UPLOADS_PREVIEW_DIR
from utils.audio_utils import extract_preview_segment
from fastapi import Request 
from services.stem_separator import classify_track_type

router = APIRouter()

@router.get("/health")
async def health_check():
    """Lightweight health check that doesn't load models"""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "service": "bridge-ml-api"
    }

@router.post("/analyze/hybrid")
async def analyze_song_hybrid(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav"]:
        return JSONResponse(status_code=400, content={"error": "Only MP3 or WAV files are supported."})

    # Extract extension based on the filename (real suffix)
    ext = os.path.splitext(file.filename)[-1].lower()

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as full_temp:
        contents = await file.read()
        full_temp.write(contents)
        full_temp.flush()
        file_path = full_temp.name

    # Create a separate temp file for preview output
    preview_temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    preview_path = preview_temp.name
    preview_temp.close()

    try:
        extract_preview_segment(file_path, preview_path, segment_duration_sec=20)
        result = process_audio_hybrid(request, preview_path, file_path)

        return {
            "status": "analyzed",
            "result": result
        }
    finally:
        os.remove(file_path)
        os.remove(preview_path)

@router.post("/test-energy")
async def test_energy():
 # returns dict: { 'vocals': path, 'drums': path, ... }
    test_stems = {
        "vocals": "uploads/stems/htdemucs/824cab72-382e-4f68-a5e9-be80dd0c7eef/vocals.wav",
        "drums": "uploads/stems/htdemucs/824cab72-382e-4f68-a5e9-be80dd0c7eef/drums.wav",
        "bass": "uploads/stems/htdemucs/824cab72-382e-4f68-a5e9-be80dd0c7eef/bass.wav",
        "other": "uploads/stems/htdemucs/824cab72-382e-4f68-a5e9-be80dd0c7eef/other.wav"
    }
    # 2. Heuristically classify track type
    track_type = classify_track_type(test_stems)

    return {
        "status": "test_energy",
        "track_type": track_type
    }