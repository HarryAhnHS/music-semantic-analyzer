from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from services.audio_processor import process_audio
from configs.index_configs import UPLOAD_DIR

router = APIRouter()

@router.post("/analyze")
async def analyze_song(file: UploadFile = File(...)):
    if file.content_type != "audio/mpeg":
        return JSONResponse(status_code=400, content={"error": "Only MP3 files are supported."})

    filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_audio(file_path)

    return {
        "status": "received",
        "filename": filename,
        "result": result
    }