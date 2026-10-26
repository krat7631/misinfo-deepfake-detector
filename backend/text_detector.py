"""Fake-news classification with the fine-tuned DistilBERT model (see training/train_text_model.py)."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import TEXT_MAX_LENGTH, TEXT_MODEL
from .text_preprocessing import normalize_article

_tokenizer = None
_model = None


def get_text_model():
    """Load lazily so the API starts fast and tests can run without weights."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL)
        _model.eval()
    return _tokenizer, _model


def _logits(texts: list[str]) -> torch.Tensor:
    tokenizer, model = get_text_model()
    inputs = tokenizer(list(texts), return_tensors="pt", truncation=True,
                       max_length=TEXT_MAX_LENGTH, padding=True)
    with torch.no_grad():
        return model(**inputs).logits


def fake_probability(texts: list[str]) -> list[float]:
    """P(fake) for already-normalised texts."""
    fake = get_text_model()[1].config.label2id["fake"]
    return torch.softmax(_logits(texts), dim=-1)[:, fake].tolist()


def fake_log_odds(texts: list[str]) -> list[float]:
    """log(P(fake) / P(real)). SHAP explains this: unlike probabilities, it does not saturate
    near 0 and 1, so confident predictions still get readable attributions."""
    label2id = get_text_model()[1].config.label2id
    logits = _logits(texts)
    return (logits[:, label2id["fake"]] - logits[:, label2id["real"]]).tolist()


def analyze_text(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    cleaned = normalize_article(text)
    tokenizer, _ = get_text_model()
    token_count = len(tokenizer(cleaned, truncation=False)["input_ids"])
    p_fake = fake_probability([cleaned])[0]

    return {
        "label": "fake" if p_fake >= 0.5 else "real",
        "fake_probability": round(p_fake, 4),
        "confidence": round(max(p_fake, 1 - p_fake), 4),
        "tokens_analyzed": min(token_count, TEXT_MAX_LENGTH),
        "truncated": token_count > TEXT_MAX_LENGTH,
    }
