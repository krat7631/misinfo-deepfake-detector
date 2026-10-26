# Architecture

## Components

| Path | Responsibility |
|---|---|
| `frontend/` | Next.js UI: text, image and video tabs; renders verdicts, SHAP highlights, face boxes, frame timeline and the live model card |
| `backend/main.py` | FastAPI layer: validation, upload limits, temp-file lifecycle, error → HTTP status mapping |
| `backend/text_preprocessing.py` | Strips publisher fingerprints (wire datelines, image credits, links, handles). Shared by training and inference |
| `backend/text_detector.py` | Fine-tuned DistilBERT inference, lazy-loaded |
| `backend/text_explainer.py` | SHAP partition explainer with a text masker; word-level contributions to P(fake) |
| `backend/face_crop.py` | YuNet face detection and square face crops. Shared by training, evaluation and inference |
| `backend/image_detector.py` | Fine-tuned ViT-Base inference on every detected face (up to 5) |
| `backend/video_detector.py` | Time-based frame sampling, per-frame face scoring, aggregation |
| `backend/explainer.py` | Optional LLM summary, only when `OPENAI_API_KEY` is set |
| `training/` | Reproducible training + evaluation scripts that write `reports/*.json` |

## Request flow

```
text  ──► normalize_article ──► DistilBERT ──► P(fake) ─┬─► verdict
                                                        └─► SHAP (≈0.5–2 s on CPU) ──► token contributions

image ──► stream to temp file (≤25 MB) ──► YuNet faces ──► select faces ──► 224×224 crops ──► ViT ──► P(fake) per face
                                                                                              └─► verdict = most suspicious face

video ──► sample 1 frame / 0.5 s (≤48) ──► largest face per frame ──► ViT ──► timeline
                                                                          └─► verdict = most suspicious 3-frame window
```

## Design decisions

- **One preprocessing path.** Text cleaning and face cropping live in `backend/` and are imported by
  the training scripts, so the model never sees inputs prepared differently from what it was trained on.
- **Fingerprint removal before training.** In WELFake, "(Reuters)" appears in 61% of real articles and
  almost no fake ones; image credits and tweet links appear only in fake ones. Left in, the classifier
  learns to recognise publishers. They are stripped, and duplicates are removed before splitting.
- **Out-of-domain evaluation is part of the pipeline,** not an afterthought: LIAR2 for text, DeepFakeFace
  (diffusion models) for images, plus a low-resolution variant that rules out image size as a shortcut.
- **Faces only.** The image model is trained on face crops, so it is applied only to detected faces and
  returns HTTP 422 when there are none, rather than guessing on content it was never trained for.
- **Score the faces the model was evaluated on.** The largest face is always scored; secondary faces only
  if they are ≥48 px, fully in frame and confidently detected. Scoring every face nearly doubled false
  positives on real photos (8.2% vs 4.3%), because tiny background faces look synthetic once upscaled.
- **Video verdict = most suspicious 1.5 s window.** A whole-clip mean misses a swapped-in segment inside a
  real video; the single worst frame lets one noisy frame flip the verdict. The mean, max and share of
  flagged frames are returned alongside it.
- **Both models are tested on sources they never trained on**, and the image model is trained on several
  generators: trained on StyleGAN alone, it caught 3–6% of diffusion fakes.
- **Lazy model loading** keeps startup fast and lets the test suite run with mocked models and no weights.

## Reliability and safety controls

- Upload limit `MAX_UPLOAD_BYTES`, enforced while streaming (HTTP 413) — the file is never fully buffered.
- Extension allowlists for image and video endpoints; unreadable images return HTTP 400.
- Text input capped at 20,000 characters.
- Temporary files use `NamedTemporaryFile` with random prefixes and are deleted in `finally`.
- CORS restricted to `ALLOWED_ORIGINS`, no credentials.

## Future work

- Fact-verification against retrieved sources for short claims (style-based classification does not transfer to them — see the LIAR2 results).
- Add more generators (newer diffusion and video face-swap models) to narrow the unseen-generator gap.
- Move video inference to a background job queue for long clips.
- Model versioning metadata in every response.
