"""Video analysis: score sampled frames with the image model and aggregate over time.

Frames are sampled at a fixed time interval (not a fixed frame count) so short and long
clips are treated alike, and the largest face per frame is scored.

The verdict uses the most suspicious window of WINDOW_FRAMES consecutive scored frames
(about 1.5 s at the default sampling rate), averaged. Averaging the whole clip would miss a
manipulated segment inside an otherwise real video; taking the single worst frame would let
one noisy frame flip the verdict. A short window catches sustained manipulation and dilutes
isolated spikes.
"""

import cv2

from .config import VIDEO_MAX_FRAMES, VIDEO_SAMPLE_SECONDS
from .face_crop import crop_face, detect_faces
from .image_detector import NoFaceError, fake_probability

WINDOW_FRAMES = 3


def summarize_frame_scores(scores: list[float], timestamps: list[float] | None = None) -> dict:
    """Aggregate per-frame P(fake) into a verdict. Pure function, unit tested."""
    if not scores:
        raise NoFaceError("No face detected in any sampled frame.")
    timestamps = timestamps or list(range(len(scores)))

    window = min(WINDOW_FRAMES, len(scores))
    means = [sum(scores[i:i + window]) / window for i in range(len(scores) - window + 1)]
    start = max(range(len(means)), key=means.__getitem__)
    peak = means[start]
    return {
        "label": "fake" if peak >= 0.5 else "real",
        "fake_probability": round(peak, 4),
        "most_suspicious_segment": {"start": timestamps[start], "end": timestamps[start + window - 1]},
        "mean_frame_probability": round(sum(scores) / len(scores), 4),
        "max_frame_probability": round(max(scores), 4),
        "share_of_frames_flagged": round(sum(s >= 0.5 for s in scores) / len(scores), 4),
    }


def analyze_video(file_path: str) -> dict:
    if not file_path:
        raise ValueError("Video path is required.")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video file.")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps * VIDEO_SAMPLE_SECONDS))

        timestamps, crops, sampled, frame_index = [], [], 0, 0
        while sampled < VIDEO_MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                sampled += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detect_faces(rgb)
                if faces:
                    crops.append(crop_face(rgb, faces[0]))
                    timestamps.append(round(frame_index / fps, 2))
            frame_index += 1
    finally:
        cap.release()

    scores = fake_probability(crops) if crops else []
    return {
        **summarize_frame_scores(scores, timestamps),
        "frames_sampled": sampled,
        "frames_with_face": len(scores),
        "duration_seconds": round(total_frames / fps, 2) if total_frames else None,
        "timeline": [{"t": t, "fake_probability": round(s, 4)}
                     for t, s in zip(timestamps, scores, strict=True)],
    }
