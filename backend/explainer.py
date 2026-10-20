"""Optional plain-language summary from an LLM.

Disabled unless OPENAI_API_KEY is set. The classifier's verdict and the SHAP attributions
never depend on it; it only rephrases why a text might be misleading.
"""

import logging

from .config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)


def llm_explanations_enabled() -> bool:
    return bool(OPENAI_API_KEY)


def explain_result(input_text: str) -> str:
    if not input_text or not input_text.strip():
        return "No explanation available for empty input."

    from openai import OpenAI

    prompt = (
        "You are a careful fact-checking assistant. In three sentences or fewer, point out "
        "features of the following text that commonly appear in misinformation (sourcing, "
        "emotive framing, unverifiable claims). Do not state whether it is true.\n\n"
        f"{input_text.strip()[:4000]}"
    )
    try:
        response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("OpenAI explanation failed")
        return "Explanation service unavailable."
