"""Fine-tune DistilBERT for fake-news classification and evaluate it honestly.

Pipeline
1. Load WELFake (72k articles from four sources), strip publisher fingerprints, and drop
   duplicates *before* splitting so no article appears in both train and test.
2. Stratified 80/10/10 split with a fixed seed.
3. Train a TF-IDF + logistic regression baseline on the same training rows.
4. Fine-tune distilbert-base-uncased.
5. Evaluate both on the held-out WELFake test set (in-domain) and on LIAR2 short political
   claims (out-of-domain), and write everything to reports/text_model_report.json.

Labels follow WELFake: 1 = fake, 0 = real. Fake is the positive class for precision/recall.

Usage:
    python training/train_text_model.py               # full run
    python training/train_text_model.py --smoke-test  # 2-minute pipeline check
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.text_preprocessing import build_model_input  # noqa: E402

BASE_MODEL = "distilbert-base-uncased"
ID2LABEL = {0: "real", 1: "fake"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# LIAR2 uses a six-point truthfulness scale. The clear ends map onto the binary task;
# "half-true" is ambiguous and is excluded rather than forced into either class.
LIAR2_TO_BINARY = {0: 1, 1: 1, 2: 1, 4: 0, 5: 0}  # pants-fire, false, barely-true | mostly-true, true


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "text-distilbert-welfake")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "text_model_report.json")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny run to check the pipeline end to end.")
    return parser.parse_args()


def load_welfake() -> tuple[pd.DataFrame, dict]:
    raw = load_dataset("davanstrien/WELFake", split="train").to_pandas()
    raw = raw.fillna({"title": "", "text": ""})
    raw["text_clean"] = [build_model_input(t, b) for t, b in zip(raw["title"], raw["text"], strict=True)]

    too_short = raw["text_clean"].str.len() < 20
    deduped = raw[~too_short].drop_duplicates(subset="text_clean")
    stats = {
        "raw_rows": len(raw),
        "dropped_too_short": int(too_short.sum()),
        "dropped_duplicates": int((~too_short).sum() - len(deduped)),
        "rows_after_cleaning": len(deduped),
    }
    return deduped[["text_clean", "label"]].rename(columns={"text_clean": "text"}).reset_index(drop=True), stats


def load_liar2_binary() -> pd.DataFrame:
    test = load_dataset("chengxuphd/liar2", split="test").to_pandas()
    test = test[test["label"].isin(LIAR2_TO_BINARY)]
    return pd.DataFrame({
        "text": [build_model_input(None, s) for s in test["statement"]],
        "label": test["label"].map(LIAR2_TO_BINARY).astype(int),
    }).reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample.reset_index(drop=True)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_fake: np.ndarray) -> dict:
    return {
        "samples": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_fake": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_fake": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_fake": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, p_fake), 4),
        # rows = true [real, fake], columns = predicted [real, fake]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def run_baseline(train: pd.DataFrame, eval_sets: dict[str, pd.DataFrame], seed: int) -> dict:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True, min_df=2)
    clf = LogisticRegression(max_iter=2_000, C=4.0, random_state=seed)
    clf.fit(vectorizer.fit_transform(train["text"]), train["label"])

    results = {}
    for name, df in eval_sets.items():
        p_fake = clf.predict_proba(vectorizer.transform(df["text"]))[:, 1]
        results[name] = classification_metrics(df["label"].to_numpy(), (p_fake >= 0.5).astype(int), p_fake)
    return results


def tokenize(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df.rename(columns={"label": "labels"}), preserve_index=False)
    return ds.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
                  batched=True, remove_columns=["text"])


def predict_proba(trainer: Trainer, ds: Dataset) -> np.ndarray:
    logits = trainer.predict(ds).predictions
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
    if args.smoke_test:
        args.train_size, args.val_size, args.epochs = 400, 200, 1
        args.output_dir = ROOT / "models" / "smoke-test"
        args.report = ROOT / "reports" / "smoke_test_report.json"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()

    print("Loading and cleaning WELFake…", flush=True)
    data, cleaning_stats = load_welfake()
    train_full, rest = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=args.seed)
    val_full, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=args.seed)
    train = stratified_sample(train_full, args.train_size, args.seed)
    val = stratified_sample(val_full, args.val_size, args.seed)
    liar2 = load_liar2_binary()
    if args.smoke_test:
        test = stratified_sample(test, 300, args.seed)
        liar2 = stratified_sample(liar2, 300, args.seed)
    eval_sets = {"welfake_test": test.reset_index(drop=True), "liar2_test_ood": liar2}
    print(f"train {len(train)} | val {len(val)} | test {len(test)} | LIAR2 {len(liar2)}", flush=True)

    print("Training TF-IDF + logistic regression baseline…", flush=True)
    baseline = run_baseline(train, eval_sets, args.seed)
    print(json.dumps(baseline, indent=2), flush=True)

    print(f"Fine-tuning {BASE_MODEL}…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_fake",
        logging_steps=50,
        seed=args.seed,
        dataloader_pin_memory=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenize(train, tokenizer, args.max_length),
        eval_dataset=tokenize(val, tokenizer, args.max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_eval_metrics,
    )
    train_started = time.time()
    trainer.train()
    train_minutes = (time.time() - train_started) / 60

    distilbert = {}
    for name, df in eval_sets.items():
        p_fake = predict_proba(trainer, tokenize(df, tokenizer, args.max_length))
        distilbert[name] = classification_metrics(df["label"].to_numpy(), (p_fake >= 0.5).astype(int), p_fake)
    print(json.dumps(distilbert, indent=2), flush=True)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    report = {
        "task": "binary fake-news classification (1 = fake, 0 = real)",
        "base_model": BASE_MODEL,
        "datasets": {
            "train_and_in_domain_test": "davanstrien/WELFake",
            "out_of_domain_test": "chengxuphd/liar2 (test split; half-true excluded; "
                                  "pants-fire/false/barely-true = fake, mostly-true/true = real)",
        },
        "cleaning": cleaning_stats,
        "split": {"seed": args.seed, "train_pool": len(train_full), "train_used": len(train),
                  "val_used": len(val), "test": len(test), "liar2_test": len(liar2)},
        "hyperparameters": {"max_length": args.max_length, "epochs": args.epochs,
                            "batch_size": args.batch_size, "learning_rate": args.learning_rate},
        "results": {"tfidf_logreg_baseline": baseline, "distilbert": distilbert},
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
