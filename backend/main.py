from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from text_detector import analyze_text
from image_detector import analyze_image # corrected function name
from video_detector import analyze_video
from explainer import explain_result
import shutil
import os

app = FastAPI()

# Enable CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-text")
async def analyze_text_endpoint(text: str = Form(...)):
    result = analyze_text(text)
    explanation = explain_result(text)
    return {
        "result": result,
        "explanation": explanation
    }

@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_image(file_path)

    os.remove(file_path)  # Clean up
    return result

@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video(file_path)

    os.remove(file_path)  # Clean up
    return result

