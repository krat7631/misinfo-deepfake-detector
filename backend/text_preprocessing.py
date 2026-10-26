"""Text normalisation shared by training and inference.

News datasets leak their source through boilerplate: real articles carry wire-service
datelines such as "WASHINGTON (Reuters) -", while fake ones end with "Featured image via
Getty Images" or embedded tweet links. A model trained on raw text learns to recognise the
publisher instead of the content. These patterns are stripped before training and before
every prediction, so both sides see the same input distribution.
"""

import re

_SOURCE_PATTERNS = [
    # Wire-service datelines: "WASHINGTON (Reuters) - ", "LONDON/PARIS (Reuters) -"
    re.compile(r"^[^\n]{0,80}?\((?:Reuters|AP|AFP)\)\s*[-–—]?\s*", re.IGNORECASE),
    re.compile(r"\((?:Reuters|AP|AFP)\)", re.IGNORECASE),
    # Image credits and attribution lines
    re.compile(r"Featured image[^\n]*", re.IGNORECASE),
    re.compile(r"\b(?:photo|image)s?\s+(?:by|via|credit:?)[^\n.]*", re.IGNORECASE),
    re.compile(r"\bGetty Images\b", re.IGNORECASE),
    re.compile(r"21st Century Wire", re.IGNORECASE),
    # Links and embedded social media
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    re.compile(r"pic\.twitter\.com/\S+", re.IGNORECASE),
    re.compile(r"@\w+"),
]
_WHITESPACE = re.compile(r"\s+")


def normalize_article(text: str) -> str:
    """Remove publisher fingerprints and collapse whitespace."""
    for pattern in _SOURCE_PATTERNS:
        text = pattern.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def build_model_input(title: str | None, body: str | None) -> str:
    """Join title and body the way the model was trained on them."""
    parts = [part.strip() for part in (title or "", body or "") if part and part.strip()]
    return normalize_article(" ".join(parts))
