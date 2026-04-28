# Frontend (Next.js)

This folder contains the web UI for the AI misinformation and deepfake detector.
The frontend lets you:

- submit text for misinformation analysis
- upload images for deepfake-related face analysis
- upload videos for heuristic suspicious-frame checks
- view structured, readable results from the backend API

---

## Quick Summary (for beginners)

You need to run **two apps**:

1. **Backend (FastAPI)** at `http://localhost:8000`
2. **Frontend (Next.js, this folder)** at `http://localhost:3000`

The frontend talks to the backend using `NEXT_PUBLIC_API_BASE_URL`.
If backend is not running, frontend forms will show API/network errors.

---

## Prerequisites

Before starting the frontend, install:

- Node.js 18+ (Node 20 recommended)
- npm (comes with Node)
- Backend dependencies from the project root (`requirements.txt`)

To verify Node is installed:

```bash
node -v
npm -v
```

---

## Step-by-Step Setup

Run the following commands from the `frontend` directory:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

The app will start at [http://localhost:3000](http://localhost:3000).

---

## How to Run the Full Codebase

Use two terminals.

### Terminal A: backend (from repo root)

```bash
source venv311/bin/activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Check backend health:

```bash
curl http://localhost:8000/health
```

### Terminal B: frontend (from `frontend/`)

```bash
npm run dev
```

Open `http://localhost:3000` and test:

- text input -> **Analyze Text**
- image upload -> **Analyze Image**
- video upload -> **Analyze Video**

---

## Environment Variables

### Frontend (`frontend/.env.local`)

- `NEXT_PUBLIC_API_BASE_URL`: backend base URL  
  Default: `http://localhost:8000`

Example:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### LLM / ChatGPT explanation key (backend side)

If you want LLM-generated explanations, set your own API key in the backend `.env`:

- `OPENAI_API_KEY=...`

Without this key, core analysis endpoints still run; only explanation generation is disabled.
For full backend setup details, see the root `README.md`.

---

## Main Screen Features

- Text analysis form (`/analyze-text`)
- Image analysis upload (`/analyze-image`)
- Video analysis upload (`/analyze-video`)
- Styled result cards + raw JSON viewer for debugging

---

## Common Errors and Fixes

### `localStorage.getItem is not a function` (dev 500)

Some environments (including Cursor) can start Node with a broken `--localstorage-file` flag.
That breaks Next.js dev overlay.

This repo already includes a fix in `npm run dev`:

- it starts Node with `--localstorage-file=.next/dev-localstorage.json`

If you still see the error:

1. check Cursor settings for `NODE_OPTIONS`
2. remove invalid `--localstorage-file` values
3. restart terminal and run `npm run dev` again

### Frontend loads but requests fail

Usually backend is not running or API URL is wrong.

Check:

1. backend terminal is active on port `8000`
2. `NEXT_PUBLIC_API_BASE_URL` points to that backend
3. CORS settings allow your frontend origin

### Running frontend from LAN URL

If accessing from another device and Next warns about cross-origin in dev:

```bash
NEXT_DEV_EXTRA_ORIGINS=http://192.168.x.x:3000
```

Use your machine IP and actual dev port.

---

## Notes for Demo / LinkedIn

For a clean demo:

- keep backend and frontend both running
- show one text + one image run in a short screen recording
- mention that LLM explanations require a user-provided API key
- mention deepfake video check is currently heuristic (not production forensic grade)
