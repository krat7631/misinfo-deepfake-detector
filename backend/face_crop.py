"""Face detection and cropping shared by training, evaluation and the API.

The image classifier only ever sees a square crop around a detected face, resized to the
model's input size. Using the same crop everywhere matters: if training images were tight
face crops but uploads were full photos, the model would be judging framing, not faces.

Detector: OpenCV YuNet (MIT licence), bundled in backend/assets.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

YUNET_PATH = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
MIN_SCORE = 0.6
MARGIN = 1.4  # crop side = longest box side * MARGIN, so hairline and jaw are included
CROP_SIZE = 224

_detector = None


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    score: float

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height,
                "score": round(self.score, 3)}


def _get_detector():
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), MIN_SCORE)
    return _detector


def detect_faces(rgb: np.ndarray) -> list[FaceBox]:
    """Return detected faces, largest first. Expects an RGB uint8 array."""
    height, width = rgb.shape[:2]
    detector = _get_detector()
    detector.setInputSize((width, height))
    _, faces = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if faces is None:
        return []
    boxes = [FaceBox(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])) for f in faces]
    return sorted(boxes, key=lambda b: b.width * b.height, reverse=True)


def crop_face(rgb: np.ndarray, box: FaceBox, size: int = CROP_SIZE) -> Image.Image:
    """Square crop around the face box, shifted (never padded) to stay inside the frame.

    Padding would add stretched-pixel streaks that only appear in some datasets, which a
    classifier can learn as a shortcut.
    """
    height, width = rgb.shape[:2]
    side = min(int(max(box.width, box.height) * MARGIN), height, width)
    cx, cy = box.x + box.width // 2, box.y + box.height // 2
    left = min(max(cx - side // 2, 0), width - side)
    top = min(max(cy - side // 2, 0), height - side)

    crop = rgb[top:top + side, left:left + side]
    return Image.fromarray(crop).resize((size, size), Image.Resampling.BICUBIC)


def largest_face_crop(image: Image.Image, size: int = CROP_SIZE) -> tuple[Image.Image, FaceBox] | None:
    """Crop the most prominent face, or None when no face is found."""
    rgb = np.asarray(image.convert("RGB"))
    faces = detect_faces(rgb)
    if not faces:
        return None
    return crop_face(rgb, faces[0], size), faces[0]
