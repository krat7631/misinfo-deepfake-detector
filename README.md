# AI-Powered Misinformation and Deepfake Detector

End-to-end project with a FastAPI backend and Next.js frontend for:

- text misinformation classification
- image analysis with DeepFace outputs
- video suspicious-frame heuristic checks
- optional LLM explanation generation

This README is written for beginners so you can run everything locally without guessing.

---

## What This Project Demonstrates

- real model-backed API inference
- full-stack integration (Python backend + React/Next frontend)
- safer file upload handling and validation
- practical engineering workflow (tests, linting, CI)
- developer documentation and troubleshooting

---

## Beginner-Friendly Overview

You run **two servers**:

1. Backend API at `http://localhost:8000`
2. Frontend UI at `http://localhost:3000`

The frontend sends text/image/video requests to the backend.
If backend is down, frontend requests fail (this is expected).

---

## Tech Stack

- **Backend:** FastAPI, Python, PyTorch, Transformers, DeepFace, OpenCV
- **Frontend:** Next.js 15, React, TypeScript
- **Optional LLM Explanation:** OpenAI API key via backend environment

---

## Prerequisites

Install these first:

- Python 3.11 (recommended for this repository's dependency set)
- Node.js 18+ (Node 20 recommended)
- npm

Check versions:

```bash
python3 --version
node -v
npm -v
```

---

## Quick Start (End-to-End)

### 1) Install backend dependencies

From project root:

```bash
python3 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

### 2) Configure backend environment

```bash
cp .env.example .env
```

If you want LLM explanations, add your own key:

```bash
OPENAI_API_KEY=your_key_here
```

Without this key, core analysis still works; only explanation text is disabled.

### 3) Start backend

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

### 4) Start frontend (new terminal)

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open:

- `http://localhost:3000`

If needed, set the frontend API URL in `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## How to Use the App

On the main page:

1. **Analyze text:** paste a headline/paragraph and click Analyze
2. **Analyze image:** upload supported image and run analysis
3. **Analyze video:** upload supported video and run heuristic scan

You will see a styled result summary plus raw JSON for debugging.

---

## Project Structure

```text
.
├── backend/
├── frontend/
├── docs/
├── scripts/
├── tests/
├── data/
├── reports/
├── .env.example
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quality Checks

From project root:

```bash
make lint
make test
```

Optional evaluation:

```bash
make eval
```

This runs `scripts/evaluate_text_detector.py` and writes results to `reports/evaluation_results.json`.

---

## Common Troubleshooting

### Backend import/install errors

- use Python 3.11
- ensure your virtual environment is activated
- re-run `pip install -r requirements.txt`

### Frontend starts but API requests fail

- confirm backend is running on port 8000
- check `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local`
- check CORS config in backend if using non-localhost origins

### Next.js dev error about `localStorage`

This repo uses a safer dev command that provides a valid `--localstorage-file` path.
If you still see issues, clear invalid `NODE_OPTIONS` flags and restart terminal.

---

## Security Notes

- Do not commit `.env` files or API keys
- Upload checks are extension + size based; always validate data handling in production deployments
- Video detection in this project is heuristic and should not be treated as forensic proof

---

## LinkedIn Project Positioning (Suggested)

To present this project strongly:

- highlight full-stack AI integration
- show one text + image + video run in demo
- mention robust error handling and test/CI setup
- clearly state LLM explanations require user-provided API key

---

## Additional Docs

- `frontend/README.md` - frontend-specific setup and troubleshooting
- `docs/ARCHITECTURE.md` - architecture notes
- `docs/DEPLOYMENT.md` - deployment checklist
