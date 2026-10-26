"""Synthetic-face detection with the fine-tuned ViT model (see training/train_image_model.py).

Faces are cropped exactly as in training and scored independently; the image verdict follows
the most suspicious face, because one swapped face is enough to make an image manipulated.

The largest face is always scored. Other faces are scored only if they look like the faces the
model was evaluated on: at least MIN_FACE_SIDE pixels, fully inside the frame, and confidently
detected. Tiny or edge-clipped background faces are heavily upscaled by the crop and produced
false positives on real photos, so they are reported as skipped instead of scored.
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

from .config import IMAGE_MODEL
from .face_crop import FaceBox, crop_face, detect_faces

MAX_FACES = 5
MIN_FACE_SIDE = 48
MIN_SECONDARY_SCORE = 0.8
EDGE_MARGIN = 2
_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

_model = None


class NoFaceError(ValueError):
    """Raised when an image contains no detectable face to analyse."""


def get_image_model():
    global _model
    if _model is None:
        _model = ViTForImageClassification.from_pretrained(IMAGE_MODEL)
        _model.eval()
    return _model


def select_faces(faces: list[FaceBox], width: int, height: int) -> tuple[list[FaceBox], int]:
    """Largest face plus any secondary faces that are big, unclipped and confidently detected.
    Expects faces sorted largest first; returns (faces to score, number skipped)."""
    if not faces:
        return [], 0

    def scorable(box: FaceBox) -> bool:
        inside = (box.x > EDGE_MARGIN and box.y > EDGE_MARGIN
                  and box.x + box.width < width - EDGE_MARGIN and box.y + box.height < height - EDGE_MARGIN)
        return min(box.width, box.height) >= MIN_FACE_SIDE and box.score >= MIN_SECONDARY_SCORE and inside

    selected = [faces[0]] + [box for box in faces[1:] if scorable(box)][:MAX_FACES - 1]
    return selected, len(faces) - len(selected)


def fake_probability(crops: list[Image.Image], batch_size: int = 16) -> list[float]:
    model = get_image_model()
    fake_index = model.config.label2id["fake"]
    scores: list[float] = []
    for start in range(0, len(crops), batch_size):
        batch = torch.stack([_TO_TENSOR(c) for c in crops[start:start + batch_size]])
        with torch.no_grad():
            probs = torch.softmax(model(pixel_values=batch).logits, dim=-1)
        scores.extend(probs[:, fake_index].tolist())
    return scores


def analyze_image(image_path: str) -> dict:
    if not image_path:
        raise ValueError("Image path is required.")

    try:
        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"))
    except OSError as exc:
        raise ValueError("File is not a readable image.") from exc

    faces, skipped = select_faces(detect_faces(rgb), width=rgb.shape[1], height=rgb.shape[0])
    if not faces:
        raise NoFaceError("No face detected. The image model only analyses faces.")

    scores = fake_probability([crop_face(rgb, box) for box in faces])
    top = max(scores)
    return {
        "label": "fake" if top >= 0.5 else "real",
        "fake_probability": round(top, 4),
        "faces_analyzed": len(faces),
        "faces_skipped": skipped,
        "faces": [{"box": box.as_dict(), "fake_probability": round(score, 4)}
                  for box, score in zip(faces, scores, strict=True)],
        "image_size": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
    }
