# Architecture Overview

## System Components

- `frontend/`: Next.js user interface
- `backend/main.py`: FastAPI API layer and request validation
- `backend/text_detector.py`: text classification model inference
- `backend/image_detector.py`: image deepfake heuristics
- `backend/video_detector.py`: sampled frame-based video analysis
- `backend/explainer.py`: LLM explanation generation

## Request Flow

1. User submits text or media.
2. FastAPI endpoint validates request.
3. For media, file is streamed to a secure temporary file.
4. Detector module performs inference.
5. Optional explanation service generates human-readable rationale.
6. API returns structured JSON.

## Reliability and Safety Controls

- Upload max size controlled by `MAX_UPLOAD_BYTES`
- File extension allowlists for image/video endpoints
- Temporary files created with `NamedTemporaryFile` and cleaned in `finally`
- Typed exceptions translated into HTTP status codes

## Performance Notes

- Text model is lazily loaded in `text_detector` on first request.
- Video detector samples frames (`FRAME_SAMPLE_INTERVAL`) to reduce compute.
- Further production scaling should move expensive inference to background workers.

## Suggested Future Evolution

- Separate API routers/services/schemas for cleaner modularity
- Introduce queue-based async processing for video jobs
- Add monitoring stack (latency/error dashboards)
- Add model versioning metadata in responses
