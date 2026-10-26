import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import REPORTS_DIR
from .explainer import explain_result, llm_explanations_enabled
from .image_detector import NoFaceError, analyze_image
from .text_detector import analyze_text
from .text_explainer import explain_text
from .video_detector import analyze_video

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_TEXT_CHARS = 20_000
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "backend/temp"))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI(title="Misinformation & Deepfake Detector", version="2.0.0")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
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


def _run_media_analysis(analyze, file_path: Path, media_type: str) -> dict:
    try:
        return analyze(str(file_path))
    except NoFaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("%s analysis runtime failure", media_type)
        raise HTTPException(status_code=500, detail=f"{media_type.capitalize()} analysis failed.") from exc
    finally:
        file_path.unlink(missing_ok=True)


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.get("/model-card")
async def model_card():
    """Evaluation results committed in reports/, so the UI can show real numbers."""
    cards = {}
    for name in ("text_model_report", "image_model_report"):
        path = REPORTS_DIR / f"{name}.json"
        if path.exists():
            report = json.loads(path.read_text())
            cards[name] = {"base_model": report["base_model"], "datasets": report["datasets"],
                           "results": report["results"],
                           "generators_seen_in_training": report.get("generators_seen_in_training")}
    return {"models": cards, "llm_explanations_enabled": llm_explanations_enabled()}


@app.post("/analyze-text")
async def analyze_text_endpoint(text: str = Form(...), skip_explanation: bool = Form(False)):
    # Form booleans follow checkbox semantics (absent = False), so the opt-out is the flag.
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Text is limited to {MAX_TEXT_CHARS} characters.")

    response = {"result": analyze_text(text)}
    if not skip_explanation:
        response["attribution"] = explain_text(text)
    if llm_explanations_enabled():
        response["explanation"] = explain_result(text)
    return response


@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    suffix = _validate_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS, "image")
    file_path = await _save_upload_to_temp(file, suffix)
    return _run_media_analysis(analyze_image, file_path, "image")


@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    suffix = _validate_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS, "video")
    file_path = await _save_upload_to_temp(file, suffix)
    return _run_media_analysis(analyze_video, file_path, "video")
