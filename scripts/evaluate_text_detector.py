import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from backend.text_detector import analyze_text


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(label: str) -> str:
    return label.strip().lower()


def evaluate(samples: list[dict]) -> dict:
    y_true = []
    y_pred = []

    for sample in samples:
        text = sample["text"]
        expected = normalize_label(sample["label"])
        pred = normalize_label(analyze_text(text)["label"])
        y_true.append(expected)
        y_pred.append(pred)

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "samples": len(samples),
        "labels": labels,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate text misinformation detector.")
    parser.add_argument(
        "--dataset",
        default="data/text_eval_samples.json",
        help="Path to evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        default="reports/evaluation_results.json",
        help="Path to store evaluation report JSON.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    samples = load_dataset(dataset_path)
    results = evaluate(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation report to {output_path}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
