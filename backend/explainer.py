import logging
import os

logger = logging.getLogger(__name__)

def _get_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not configured.")
    return OpenAI(api_key=api_key)

def explain_result(input_text: str) -> str:
    if not input_text or not input_text.strip():
        return "No explanation available for empty input."

    prompt = (
        "You are a misinformation expert. "
        "Explain in simple terms why the following content may be misinformation:\n\n"
        f"{input_text.strip()}"
    )

    try:
        client = _get_client()
    except RuntimeError:
        return "OpenAI explanation unavailable because OPENAI_API_KEY is not configured."

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.exception("OpenAI explanation failed")
        return f"Explanation service unavailable: {exc}"

# Test block
if __name__ == "__main__":
    sample_text = "The COVID-19 vaccine implants microchips in people."
    explanation = explain_result(sample_text)
    print("🧠 GPT Explanation:\n", explanation)

