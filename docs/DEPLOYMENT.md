# Deployment Guide

## Prerequisites

- Python 3.11+
- API key for OpenAI (optional but needed for explanations)
- Linux server or PaaS runtime

## Environment Variables

Copy `.env.example` and set values:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `ALLOWED_ORIGINS`
- `MAX_UPLOAD_BYTES`
- `UPLOAD_DIR`

## Local Production-Like Run

```bash
pip install -r requirements.txt
gunicorn -w 2 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000
```

## Reverse Proxy Suggestions

- Put Nginx or equivalent in front of FastAPI
- Enforce HTTPS
- Set request body size limit aligned to `MAX_UPLOAD_BYTES`
- Add rate limits per IP

## Observability Starter Checklist

- Add request and error logs with request IDs
- Track p50/p95 latency for each endpoint
- Alert on elevated 5xx errors

## Release Checklist

- `make lint`
- `make test`
- `pip-audit -r requirements.txt`
- regenerate `reports/evaluation_results.json`
- update README metrics section
