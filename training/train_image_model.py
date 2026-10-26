"""Fine-tune a Vision Transformer to detect synthetic faces, and test how far it generalises.

Pipeline
1. Face-crop every image with the same YuNet crop the API uses (backend/face_crop.py).
2. Fine-tune google/vit-base-patch16-224-in21k on
   - 140k Real-and-Fake-Faces (real: FFHQ photos, fake: StyleGAN), and
   - DeepFakeFace identity folders 10-29 (real: IMDB-WIKI photos, fake: Stable Diffusion
     inpainting and InsightFace face swaps; see training/download_dff_sample.py),
   with JPEG and downscale augmentation so the model cannot rely on image quality alone.
   Stable Diffusion text-to-image is never trained on: it is the unseen-generator test.
3. Evaluate on
   - the held-out 140k test images;
   - DeepFakeFace identity folders 00-03, people never seen in training: inpainting and face
     swaps (seen generators) and text-to-image (unseen generator);
   - DeepFakeFace again with every crop reduced to a common low resolution, which removes
     the resolution gap between DFF's real and fake images as an explanation.
4. Write everything to reports/image_model_report.json.

`--no-dff-train` reproduces the first, GAN-only model (reports/image_model_report_v1_gan_only.json),
which caught only 3-6% of diffusion fakes and motivated adding them to training.

Labels: 1 = fake, 0 = real. Fake is the positive class.

Usage:
    python training/download_dff_sample.py --split test && python training/download_dff_sample.py --split train
    python training/train_image_model.py               # full run
    python training/train_image_model.py --smoke-test  # pipeline check on a few hundred images
"""

import argparse
import io
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Trainer, TrainingArguments, ViTForImageClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.face_crop import CROP_SIZE, largest_face_crop  # noqa: E402

BASE_MODEL = "google/vit-base-patch16-224-in21k"
FACES_REPO = "TheKernel01/140k-Real-and-Fake-Faces"
FACES_FILES = {
    "train": "data/train-00000-of-00006.parquet",
    "val": "data/validation-00000-of-00002.parquet",
    "test": "data/test-00000-of-00002.parquet",
}
DFF_TEST_DIR = ROOT / ".hf_cache" / "dff_sample"
DFF_TRAIN_DIR = ROOT / ".hf_cache" / "dff_train"
DFF_FAKES = ["inpainting", "insight", "text2img"]
DFF_TRAIN_SOURCES = {"wiki": 0, "inpainting": 1, "insight": 1}
DFF_VAL_FRACTION = 0.1
ID2LABEL = {0: "real", 1: "fake"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
# ViT-in21k was pre-trained with mean = std = 0.5 per channel.
NORMALIZE = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
LOW_RES = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-size", type=int, default=12_000, help="140k-faces training images")
    parser.add_argument("--no-dff-train", action="store_true", help="Train on GAN faces only (v1 model)")
    parser.add_argument("--val-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "image-vit-faces")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "image_model_report.json")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


class RandomJPEG:
    """Re-encode at a random JPEG quality, so compression level is not a usable signal."""

    def __init__(self, quality=(55, 95), p=0.5):
        self.quality, self.p = quality, p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        return Image.open(buffer).convert("RGB")


class RandomDownscale:
    """Downscale then upscale, simulating low-resolution sources such as old photos."""

    def __init__(self, sizes=(56, 160), p=0.3):
        self.sizes, self.p = sizes, p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        side = random.randint(*self.sizes)
        return image.resize((side, side), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BICUBIC)


TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(CROP_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(),
    RandomJPEG(),
    RandomDownscale(),
    transforms.ToTensor(),
    NORMALIZE,
])
EVAL_TRANSFORM = transforms.Compose([transforms.ToTensor(), NORMALIZE])


def low_res(image: Image.Image) -> Image.Image:
    return image.resize((LOW_RES, LOW_RES), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BICUBIC)


class CropDataset(Dataset):
    def __init__(self, crops: list[Image.Image], labels: list[int], transform):
        self.crops, self.labels, self.transform = crops, labels, transform

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> dict:
        return {"pixel_values": self.transform(self.crops[idx]), "labels": self.labels[idx]}


def crop_all(images, labels) -> tuple[list[Image.Image], list[int], int]:
    crops, kept, missed = [], [], 0
    for image, label in zip(images, labels, strict=True):
        found = largest_face_crop(image)
        if found is None:
            missed += 1
            continue
        crops.append(found[0])
        kept.append(label)
    return crops, kept, missed


def load_faces_split(split: str, n: int, seed: int) -> tuple[list[Image.Image], list[int], int]:
    table = pq.read_table(hf_hub_download(FACES_REPO, FACES_FILES[split], repo_type="dataset"))
    rows = table.to_pylist()
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    images = (Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows)
    return crop_all(images, [int(r["label"]) for r in rows])


def load_dff_test(limit: int | None) -> tuple[dict[str, tuple[list, list]], dict[str, int]]:
    """Return per-source crops; the real set is shared across the three fake generators."""
    sets, missed = {}, {}
    for source in ["wiki", *DFF_FAKES]:
        files = sorted((DFF_TEST_DIR / source).rglob("*.jpg"))[:limit]
        label = 0 if source == "wiki" else 1
        crops, labels, miss = crop_all((Image.open(p).convert("RGB") for p in files), [label] * len(files))
        sets[source], missed[source] = (crops, labels), miss
    return sets, missed


def load_dff_train(limit: int | None, seed: int) -> tuple[tuple[list, list], tuple[list, list], dict[str, int]]:
    """Crops from the training identity folders, split into train and a small validation share."""
    crops, labels, missed = [], [], {}
    for source, label in DFF_TRAIN_SOURCES.items():
        files = sorted((DFF_TRAIN_DIR / source).rglob("*.jpg"))[:limit]
        c, y, missed[source] = crop_all((Image.open(p).convert("RGB") for p in files), [label] * len(files))
        crops += c
        labels += y
    order = list(range(len(crops)))
    random.Random(seed).shuffle(order)
    n_val = int(len(order) * DFF_VAL_FRACTION)
    pick = lambda idx: ([crops[i] for i in idx], [labels[i] for i in idx])  # noqa: E731
    return pick(order[n_val:]), pick(order[:n_val]), missed


def metrics(y_true: np.ndarray, p_fake: np.ndarray) -> dict:
    y_pred = (p_fake >= 0.5).astype(int)
    return {
        "samples": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_fake": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_fake": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_fake": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, p_fake), 4),
        # rows = true [real, fake], columns = predicted [real, fake]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def predict(trainer: Trainer, crops: list[Image.Image], labels: list[int], transform=EVAL_TRANSFORM) -> np.ndarray:
    logits = trainer.predict(CropDataset(crops, labels, transform)).predictions
    return torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()


def compute_eval_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1_fake": f1_score(labels, preds)}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    dff_limit = None
    if args.smoke_test:
        args.train_size, args.val_size, args.test_size, args.epochs = 320, 100, 200, 1
        args.output_dir = ROOT / "models" / "smoke-test-image"
        args.report = ROOT / "reports" / "smoke_test_image_report.json"
        dff_limit = 100

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()

    print("Face-cropping 140k Real-and-Fake-Faces…", flush=True)
    train_x, train_y, train_miss = load_faces_split("train", args.train_size, args.seed)
    val_x, val_y, val_miss = load_faces_split("val", args.val_size, args.seed)
    test_x, test_y, test_miss = load_faces_split("test", args.test_size, args.seed)
    print("Face-cropping DeepFakeFace test identities…", flush=True)
    dff, dff_miss = load_dff_test(dff_limit)
    dff_train_miss = {}
    if not args.no_dff_train:
        print("Face-cropping DeepFakeFace training identities…", flush=True)
        (dtx, dty), (dvx, dvy), dff_train_miss = load_dff_train(dff_limit, args.seed)
        train_x, train_y = train_x + dtx, train_y + dty
        val_x, val_y = val_x + dvx, val_y + dvy
    seen_generators = ["StyleGAN"] if args.no_dff_train else ["StyleGAN", "inpainting", "insight"]
    print(f"train {len(train_x)} (fake share {np.mean(train_y):.2f}) | val {len(val_x)} | test {len(test_x)} | "
          f"DFF test {', '.join(f'{k} {len(v[0])}' for k, v in dff.items())}", flush=True)

    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.05,
        warmup_steps=0.1,  # a fraction of total steps (transformers 5 merged warmup_ratio into this)
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_fake",
        logging_steps=50,
        seed=args.seed,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=CropDataset(train_x, train_y, TRAIN_TRANSFORM),
        eval_dataset=CropDataset(val_x, val_y, EVAL_TRANSFORM),
        compute_metrics=compute_eval_metrics,
    )
    print(f"Fine-tuning {BASE_MODEL}…", flush=True)
    train_started = time.time()
    trainer.train()
    train_minutes = (time.time() - train_started) / 60

    print("Evaluating…", flush=True)
    results = {"faces140k_test": metrics(np.array(test_y), predict(trainer, test_x, test_y))}
    real_x, real_y = dff["wiki"]
    real_p = predict(trainer, real_x, real_y)
    real_p_low = predict(trainer, [low_res(c) for c in real_x], real_y)
    for source in DFF_FAKES:
        fake_x, fake_y = dff[source]
        y = np.array(real_y + fake_y)
        results[f"dff_{source}"] = metrics(y, np.concatenate([real_p, predict(trainer, fake_x, fake_y)]))
        results[f"dff_{source}_low_res"] = metrics(
            y, np.concatenate([real_p_low, predict(trainer, [low_res(c) for c in fake_x], fake_y)])
        )
    results["dff_real_photos_flagged_fake_rate"] = round(float((real_p >= 0.5).mean()), 4)
    print(json.dumps(results, indent=2), flush=True)

    trainer.save_model(str(args.output_dir))

    report = {
        "task": "binary synthetic-face classification on face crops (1 = fake, 0 = real)",
        "base_model": BASE_MODEL,
        "datasets": {
            "faces140k": f"{FACES_REPO} (real: FFHQ, fake: StyleGAN); one parquet shard per split",
            "dff_train": None if args.no_dff_train else
                "OpenRL/DeepFakeFace identity folders 10-29 (real: IMDB-WIKI; fake: SD inpainting, InsightFace swap)",
            "dff_test": "OpenRL/DeepFakeFace identity folders 00-03, 1,000 images per source; people never "
                        "seen in training (real: IMDB-WIKI; fake: SD inpainting, InsightFace swap, SD text-to-image)",
        },
        "generators_seen_in_training": seen_generators,
        "preprocessing": {"face_detector": "OpenCV YuNet 2023mar", "crop_margin": 1.4, "crop_size": CROP_SIZE,
                          "low_res_eval_side": LOW_RES,
                          "train_augmentation": "random resized crop 0.85-1.0, horizontal flip, "
                                                "JPEG q55-95 (p=0.5), downscale 56-160px (p=0.3)"},
        "no_face_detected": {"faces140k": {"train": train_miss, "val": val_miss, "test": test_miss},
                             "dff_test": dff_miss, "dff_train": dff_train_miss},
        "split": {"seed": args.seed, "train": len(train_x), "val": len(val_x), "test": len(test_x)},
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size,
                            "learning_rate": args.learning_rate, "weight_decay": 0.05},
        "results": results,
        "training_minutes": round(train_minutes, 1),
        "total_minutes": round((time.time() - started) / 60, 1),
        "environment": {"device": str(trainer.args.device), "platform": platform.platform(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "git_commit": git_commit()},
        "smoke_test": args.smoke_test,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Report written to {args.report}", flush=True)


if __name__ == "__main__":
    main()
