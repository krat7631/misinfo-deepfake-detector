# Misinformation & Deepfake Detector

A full-stack system that classifies fake-news text and synthetic faces, explains its text predictions,
and reports how well each model holds up on data from sources it never saw in training.

- **Text:** DistilBERT fine-tuned on 20,000 WELFake articles, with SHAP word attributions.
- **Images:** ViT-Base fine-tuned on GAN and diffusion-generated faces, applied to every detected face.
- **Video:** the image model run on sampled frames, with a per-frame timeline.
- **Stack:** FastAPI · PyTorch · Hugging Face Transformers · SHAP · OpenCV · Next.js · TypeScript.

## Results at a glance

| Model | Test set | Accuracy | ROC-AUC |
|---|---|---|---|
| DistilBERT | WELFake held-out articles | 98.7% | 0.999 |
| DistilBERT | LIAR2 political claims (unseen domain) | 52.1% | 0.574 |
| ViT-Base | StyleGAN faces held-out | 98.6% | 0.999 |
| ViT-Base | Stable Diffusion inpainting, new people | 93.5% | 0.984 |
| ViT-Base | InsightFace face swaps, new people | 92.6% | 0.979 |
| ViT-Base | Stable Diffusion text-to-image (generator never trained on) | 77.6% | 0.897 |

The rows that matter most are the ones from sources the models never trained on. Details below.

## Image model

Fine-tuned `google/vit-base-patch16-224-in21k` on face crops from two datasets:

- **140k Real and Fake Faces:** 12,000 images (real FFHQ photos vs StyleGAN faces).
- **DeepFakeFace:** about 10,700 images from identity folders 10–29 (real IMDB-WIKI photos vs Stable Diffusion
  inpainting and InsightFace face swaps).

Test images come from DeepFakeFace identity folders 00–03, so every test face belongs to someone the model
never saw. Stable Diffusion text-to-image is never trained on at all.

### Why the training data includes diffusion fakes

The first version was trained on StyleGAN faces only. It reached 99.2% on held-out StyleGAN faces and
collapsed on everything else:

| Test set | v1: GAN faces only | v2: GAN + diffusion |
|---|---|---|
| StyleGAN faces held-out | 99.2% (AUC 1.000) | 98.6% (AUC 0.999) |
| SD inpainting | 49.7%, caught 4% of fakes (AUC 0.40) | **93.5%**, catches 91% (AUC 0.98) |
| InsightFace swaps | 50.7%, caught 6% (AUC 0.48) | **92.6%**, catches 89% (AUC 0.98) |
| SD text-to-image (never trained on) | 49.0%, caught 3% (AUC 0.28) | **77.6%**, catches 59% (AUC 0.90) |
| Real photos wrongly flagged | 4.7% | 4.3% |

v1 had learned StyleGAN's fingerprints, not what makes a face synthetic. An AUC below 0.5 means it rated
text-to-image fakes as *more* real than genuine photos. Adding two diffusion generators fixed the
generators it trained on and transferred partly to one it never saw: when v2 flags an unseen-generator image
as fake, it is right 93% of the time, but it still misses 41% of them.
v1's full report is kept in [`reports/image_model_report_v1_gan_only.json`](reports/image_model_report_v1_gan_only.json)
and is reproducible with `--no-dff-train`.

### Checks against shortcuts

- **Resolution.** DeepFakeFace's real photos are small (≈196×291) and its fakes are 512×512, so a model
  could separate them by sharpness alone. Training uses random JPEG re-encoding and downscaling, and every
  DeepFakeFace test is repeated with all crops reduced to 64 px: accuracy stays at 74–90%
  (see the `_low_res` rows in the report).
- **Framing.** Every image, in training and in the API, goes through the same YuNet face detector and
  square crop (`backend/face_crop.py`), so the model never compares tight crops with wide shots.

### Decisions found by testing the API

- **Which faces to score.** Scoring every detected face nearly doubled false positives on real photos
  (8.2% vs 4.3%): tiny or edge-clipped background faces, upscaled to 224 px, look synthetic. The API scores
  the largest face plus only secondary faces at least 48 px wide, fully in frame and confidently detected;
  this gives 4.7% on the same 987 real test photos.
- **How to score a video.** Averaging every frame let a real video with a swapped-in 3-second segment pass
  as real (mean 0.47). The verdict now uses the most suspicious 1.5-second window, which catches sustained
  manipulation while one noisy frame cannot flip the result.

Full report: [`reports/image_model_report.json`](reports/image_model_report.json).

## Text model

Fine-tuned `distilbert-base-uncased` on WELFake (72,134 articles from four news sources), compared against
a TF-IDF + logistic regression baseline trained on the same rows.

| Test set | Model | Accuracy | F1 (fake) | ROC-AUC |
|---|---|---|---|---|
| WELFake held-out (6,363 articles) | TF-IDF + LogReg | 95.3% | 0.949 | 0.990 |
| WELFake held-out (6,363 articles) | **DistilBERT** | **98.7%** | **0.986** | **0.999** |
| LIAR2 political claims (1,925, unseen domain) | TF-IDF + LogReg | 67.6% | 0.797 | 0.631 |
| LIAR2 political claims (1,925, unseen domain) | DistilBERT | 52.1% | 0.562 | 0.574 |

**What the numbers say.** On articles like the ones it trained on, DistilBERT makes 73% fewer errors
than the baseline (81 instead of 299). On short fact-checked political claims, neither model is useful:
69% of LIAR2 claims are false, so predicting "fake" every time would score 68.7%, and both models'
ROC-AUC is close to 0.5. The models learned the writing style of fabricated articles, and a one-line claim
such as "unemployment fell to 3.9% last month" has no style to learn from. Checking it needs evidence
retrieval, which this system does not do.

**Data hygiene that changed the result.**
- "(Reuters)" appears in 61% of real WELFake articles and 0.1% of fake ones; "Featured image", Getty
  credits and tweet links appear only in fake ones. A classifier trained on raw text learns the publisher.
  `backend/text_preprocessing.py` strips these fingerprints for training and inference alike.
- 8,457 exact duplicates were removed *before* the train/test split, so no article is in both.

Full report: [`reports/text_model_report.json`](reports/text_model_report.json).

## How it works

```
text  ──► strip publisher fingerprints ──► DistilBERT ──► P(fake) ──► SHAP word attributions
image ──► YuNet face detection ──► 224×224 face crops ──► ViT ──► P(fake) per face
video ──► 1 frame / 0.5 s (max 48) ──► largest face ──► ViT ──► timeline + most suspicious 1.5 s window
```

More detail, including design decisions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/model-card` | — | Evaluation results from `reports/` |
| POST | `/analyze-text` | form: `text`, optional `skip_explanation=true` | label, P(fake), SHAP tokens |
| POST | `/analyze-image` | multipart `file` (jpg, png, webp; ≤25 MB) | per-face boxes and P(fake); 422 if no face |
| POST | `/analyze-video` | multipart `file` (mp4, mov, avi, mkv, webm; ≤25 MB) | verdict, most suspicious segment, per-frame timeline |

## Run it locally

Requires Python 3.11 and Node 20.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install                 # API + test dependencies (CPU PyTorch)
cp .env.example .env
make run-backend             # http://localhost:8000
```

```bash
cd frontend && npm install && cp .env.example .env.local && npm run dev   # http://localhost:3000
```

The API loads models from `models/` by default (set `TEXT_MODEL` / `IMAGE_MODEL` to a local folder or a
Hugging Face Hub id). To produce them yourself:

```bash
make install-train
python training/train_text_model.py                     # ~25 min on an Apple M-series GPU
python training/download_dff_sample.py --split test     # ~170 MB sampled from DeepFakeFace
python training/download_dff_sample.py --split train    # ~570 MB
python training/train_image_model.py                    # ~40 min on an Apple M-series GPU
```

Each script writes its metrics to `reports/` and takes `--smoke-test` for a two-minute pipeline check.

## Quality checks

```bash
make lint    # ruff
make test    # pytest: API contract, upload limits, temp-file cleanup, preprocessing, video aggregation
```

CI runs lint, tests and `pip-audit` on every push. Tests mock the models, so they need no weights.

## Limitations

- The text model judges style and framing, not facts. It should not be used on short claims.
- Only faces are analysed; manipulated backgrounds, objects or captions are out of scope.
- On a generator it never trained on, the image model misses about 4 in 10 fakes. Newer generators may be harder.
- Stretched, heavily upscaled or re-compressed images raise the false-positive rate on real photos.
- The DeepFakeFace evaluation uses about 1,000 images per generator; treat differences of a point or two as noise.
- This is a portfolio research project, not a forensic tool. Do not use it as the sole basis for decisions
  about people or content.

## Data and model credits

- [WELFake](https://huggingface.co/datasets/davanstrien/WELFake) — Verma et al., 2021.
- [LIAR2](https://huggingface.co/datasets/chengxuphd/liar2) — an extended version of the LIAR fact-checking dataset.
- [140k Real and Fake Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces) — FFHQ
  (NVIDIA, CC BY-NC-SA 4.0) and StyleGAN faces; non-commercial use.
- [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) — Song et al., 2023.
- [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), [ViT-Base in21k](https://huggingface.co/google/vit-base-patch16-224-in21k), [YuNet](https://huggingface.co/opencv/face_detection_yunet) (MIT, bundled in `backend/assets/`).
