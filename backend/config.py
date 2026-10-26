"""Runtime configuration, read once from the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

# Either a local directory produced by training/ or a Hugging Face Hub model id.
TEXT_MODEL = os.getenv("TEXT_MODEL", str(ROOT / "models" / "text-distilbert-welfake"))
IMAGE_MODEL = os.getenv("IMAGE_MODEL", str(ROOT / "models" / "image-vit-faces"))
TEXT_MAX_LENGTH = 256

# Video: score one frame every VIDEO_SAMPLE_SECONDS, at most VIDEO_MAX_FRAMES frames.
VIDEO_SAMPLE_SECONDS = float(os.getenv("VIDEO_SAMPLE_SECONDS", "0.5"))
VIDEO_MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "48"))

# Optional natural-language summary from an LLM, off unless a key is configured.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

REPORTS_DIR = ROOT / "reports"
