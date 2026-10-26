"""Word-level explanations for text predictions using SHAP.

SHAP's text masker hides groups of tokens and measures how the model's output changes, giving
each token a signed contribution: positive pushes towards "fake", negative towards "real".
Contributions are in log-odds, log(P(fake) / P(real)), because probabilities saturate: when the
model is 99.9% sure, hiding any one word barely moves the probability even if it matters.

This explains what the model relied on; it is not evidence that a claim is true or false.
"""

import shap

from .config import TEXT_MAX_LENGTH
from .text_detector import fake_log_odds, get_text_model
from .text_preprocessing import normalize_article

MAX_EVALS = 300
TOP_K = 12

_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        tokenizer, _ = get_text_model()
        masker = shap.maskers.Text(tokenizer)
        _explainer = shap.Explainer(lambda texts: fake_log_odds(list(texts)), masker)
    return _explainer


def _truncate_to_model_window(text: str) -> str:
    """Cut the original string where the classifier stops reading, keeping its casing."""
    tokenizer, _ = get_text_model()
    budget = TEXT_MAX_LENGTH - 2  # [CLS] and [SEP]
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) <= budget:
        return text
    return text[:offsets[budget - 1][1]]


def explain_text(text: str) -> dict:
    window = _truncate_to_model_window(normalize_article(text))
    values = _get_explainer()([window], max_evals=MAX_EVALS, batch_size=32)
    tokens = [str(t) for t in values.data[0]]
    scores = [float(s) for s in values.values[0]]

    words = [{"token": tok, "contribution": round(score, 4)}
             for tok, score in zip(tokens, scores, strict=True) if tok.strip()]
    ranked = sorted(words, key=lambda w: abs(w["contribution"]), reverse=True)
    return {
        "method": "SHAP partition explainer on log-odds",
        "base_value": round(float(values.base_values[0]), 4),
        "tokens": words,
        "top_fake_signals": [w for w in ranked if w["contribution"] > 0][:TOP_K],
        "top_real_signals": [w for w in ranked if w["contribution"] < 0][:TOP_K],
    }
