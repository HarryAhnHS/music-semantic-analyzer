from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from services.audio_processor import process_audio
from configs.index_configs import UPLOAD_DIR, UPLOADS_PREVIEW_DIR
from utils.audio_utils import extract_preview_segment

router = APIRouter()

@router.post("/analyze")
async def analyze_song(file: UploadFile = File(...)):
    if file.content_type != "audio/mpeg":
        return JSONResponse(status_code=400, content={"error": "Only MP3 files are supported."})

    filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(UPLOAD_DIR, filename)
    # Save full upload
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate a preview for analysis (e.g., 30 seconds)
    preview_path = os.path.join(UPLOADS_PREVIEW_DIR, filename)
    extract_preview_segment(file_path, preview_path, segment_duration_sec=30)

    result = process_audio(preview_path)

    return {
        "status": "analyzed",
        "filename": filename,
        "result": result
    }