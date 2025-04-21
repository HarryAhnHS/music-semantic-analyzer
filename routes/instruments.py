# from fastapi import APIRouter, UploadFile, File
# from fastapi.responses import JSONResponse
# import shutil
# import os
# import uuid
# from services.predict_instruments import predict_instruments_from_mp3
# from configs.index_configs import UPLOAD_DIR

# router = APIRouter()

# @router.post("/predict-instruments")
# async def predict_instruments(file: UploadFile = File(...)):
#     if file.content_type != "audio/mpeg":
#         return JSONResponse(status_code=400, content={"error": "Only MP3 files are supported."})

#     filename = f"{uuid.uuid4()}.mp3"
#     file_path = os.path.join(UPLOAD_DIR, filename)

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     result = predict_instruments_from_mp3(file_path)

#     return result