# Deployment

## What the API needs

- Python 3.11, about 2 GB of RAM (both models loaded) and no GPU: inference runs on CPU.
- The two trained models, either in `models/` or on the Hugging Face Hub (set `TEXT_MODEL` and
  `IMAGE_MODEL` to the Hub ids).
- `OPENAI_API_KEY` is optional. Without it, the API works fully; only the optional LLM summary is off.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `TEXT_MODEL` | `models/text-distilbert-welfake` | Local folder or Hub id of the text classifier |
| `IMAGE_MODEL` | `models/image-vit-faces` | Local folder or Hub id of the face classifier |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated CORS origins |
| `MAX_UPLOAD_BYTES` | `26214400` (25 MB) | Upload limit, enforced while streaming |
| `UPLOAD_DIR` | `backend/temp` | Where uploads are written before analysis and deleted after |
| `VIDEO_SAMPLE_SECONDS` / `VIDEO_MAX_FRAMES` | `0.5` / `48` | Video sampling rate and cap |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | empty / `gpt-4o-mini` | Optional LLM summary |

## Production-like run

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install gunicorn
gunicorn -w 1 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000 --timeout 120
```

Use one worker per ~2 GB of RAM: each worker loads its own copy of both models. Video requests can take
several seconds on CPU, hence the longer timeout.

## Behind a reverse proxy

- Enforce HTTPS.
- Set the proxy's request-body limit to match `MAX_UPLOAD_BYTES`.
- Rate-limit `/analyze-*` per IP; SHAP and video analysis are CPU-heavy.

## Release checklist

```bash
make lint && make test
pip-audit -r requirements.txt
```

If a model is retrained, commit the new `reports/*.json` with it so `/model-card` and the README stay
in sync with the deployed weights.
