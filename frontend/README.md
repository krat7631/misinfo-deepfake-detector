# Frontend

Next.js UI for the detector. It talks to the FastAPI backend in the repository root.

## Run

```bash
cd frontend
npm install
cp .env.example .env.local   # sets NEXT_PUBLIC_API_BASE_URL, default http://localhost:8000
npm run dev
```

Open http://localhost:3000. The backend must be running (`make run-backend` from the repo root);
the status dot in the header shows whether the API is reachable.

## Structure

| Path | Purpose |
|---|---|
| `app/page.tsx` | Text / image / video tabs, submission, result routing |
| `lib/api.ts` | Typed API client and response types |
| `components/Verdict.tsx` | Shared verdict with a P(fake) meter and the 0.5 threshold |
| `components/TextResult.tsx` | SHAP token highlighting and strongest fake/real signals |
| `components/ImageResult.tsx` | Uploaded image with detected face boxes and per-face scores |
| `components/VideoResult.tsx` | Per-frame fake-probability timeline (SVG) |
| `components/ModelCard.tsx` | Evaluation results from `GET /model-card` |

## Troubleshooting

- **Requests fail:** check the backend is on port 8000 and that `ALLOWED_ORIGINS` in the backend `.env`
  includes the frontend origin.
- **`localStorage.getItem is not a function` in dev:** some editors start Node with an invalid
  `--localstorage-file` flag. `npm run dev` already passes a valid path; if it persists, clear `NODE_OPTIONS`.
- **Accessing the dev server from another device:** set `NEXT_DEV_EXTRA_ORIGINS=http://<your-ip>:3000`.
