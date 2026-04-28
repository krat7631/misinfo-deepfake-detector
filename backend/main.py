from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from .explainer import explain_result
    from .image_detector import analyze_image
    from .text_detector import analyze_text
    from .video_detector import analyze_video
except ImportError:
    from explainer import explain_result
    from image_detector import analyze_image
    from text_detector import analyze_text
    from video_detector import analyze_video
import logging
import os

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "backend/temp"))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI()
if UPLOAD_DIR.exists() and not UPLOAD_DIR.is_dir():
    UPLOAD_DIR = Path("backend/temp_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)

def _safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix else ".bin"


def _validate_extension(filename: str, allowed_extensions: set[str], media_type: str) -> str:
    suffix = _safe_suffix(filename)
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {media_type} format '{suffix}'. Allowed: {sorted(allowed_extensions)}",
        )
    return suffix


async def _save_upload_to_temp(upload_file: UploadFile, suffix: str) -> Path:
    size = 0
    with NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        prefix=f"{uuid4().hex}_",
        dir=UPLOAD_DIR,
        delete=False,
    ) as tmp:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max allowed size is {MAX_UPLOAD_BYTES} bytes.",
                )
            tmp.write(chunk)
    return Path(tmp.name)


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.post("/analyze-text")
async def analyze_text_endpoint(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = analyze_text(text)
    explanation = explain_result(text)
    return {
        "result": result,
        "explanation": explanation
    }


@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    suffix = _validate_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS, "image")
    file_path = await _save_upload_to_temp(file, suffix)

    try:
        return analyze_image(str(file_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Image analysis runtime failure")
        raise HTTPException(status_code=500, detail="Image analysis failed.") from exc
    finally:
        file_path.unlink(missing_ok=True)


@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    suffix = _validate_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS, "video")
    file_path = await _save_upload_to_temp(file, suffix)

    try:
        return analyze_video(str(file_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Video analysis runtime failure")
        raise HTTPException(status_code=500, detail="Video analysis failed.") from exc
    finally:
        file_path.unlink(missing_ok=True)

