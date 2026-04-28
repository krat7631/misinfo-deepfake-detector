"""Text classification without ``transformers.pipeline`` (avoids pulling torchvision / lzma on minimal Python builds)."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_ID = "mrm8488/bert-tiny-finetuned-fake-news-detection"
_tokenizer = None
_model = None
_LABEL_MAP = {
    # This checkpoint exposes generic labels only.
    "LABEL_0": "fake",
    "LABEL_1": "real",
}


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_ID)
        _model.eval()
    return _tokenizer, _model


def analyze_text(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    tokenizer, model = _get_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    label_id = int(torch.argmax(probs).item())
    raw_label = str(model.config.id2label[label_id])
    label = _LABEL_MAP.get(raw_label, raw_label.lower())
    score = float(probs[label_id].item())
    return {
        "label": label,
        "raw_label": raw_label,
        "score": round(score, 3),
    }
