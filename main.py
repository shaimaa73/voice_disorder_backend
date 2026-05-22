from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from app.services.model_service import predict_audio

# APP

app = FastAPI(
    title="Voice Disorder Detection API",
    description="Deep Learning API for Voice Disorder Classification",
    version="1.0.0"
)

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # رابط الفرونت 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PATHS

BASE_DIR = os.path.dirname(__file__)

UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ROUTES

@app.get("/")
def home():
    return {
        "message": "Voice Disorder Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # التحقق من نوع الملف
    if not file.filename.lower().endswith(".wav"):
        return {
            "error": "Only WAV audio files are supported"
        }

    # اسم عشوائي للملف
    unique_name = f"{uuid.uuid4()}.wav"

    temp_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        # حفظ الملف مؤقتًا
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # prediction
        result = predict_audio(temp_path)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": round(result["confidence"] * 100, 2)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        # حذف الملف بعد الانتهاء
        if os.path.exists(temp_path):
            os.remove(temp_path)