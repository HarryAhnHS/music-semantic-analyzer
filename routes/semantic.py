from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from services.audio_processor import process_audio
from configs.index_configs import UPLOAD_DIR, UPLOADS_PREVIEW_DIR
from utils.audio_utils import extract_preview_segment

from services.stem_separator import classify_track_type

router = APIRouter()

@router.post("/ttmrpp")
async def ttmrpp(file: UploadFile = File(...)):
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav"]:
        return JSONResponse(status_code=400, content={"error": "Only MP3 or WAV files are supported."})

    return {
        "status": "ttmrpp"
    }

@router.post("/analyze")
async def analyze_song(file: UploadFile = File(...)):
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav"]:
        return JSONResponse(status_code=400, content={"error": "Only MP3 or WAV files are supported."})

    ext = os.path.splitext(file.filename)[-1].lower()  # ".mp3", ".wav", etc.
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    # Save full upload
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate a preview for analysis (e.g., 30 seconds)
    preview_path = os.path.join(UPLOADS_PREVIEW_DIR, filename)
    extract_preview_segment(file_path, preview_path, segment_duration_sec=20)

    result = process_audio(preview_path, file_path)

    return {
        "status": "analyzed",
        "filename": filename,
        "result": result
    }

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