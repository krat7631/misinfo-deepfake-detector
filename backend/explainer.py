import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load the API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def explain_result(input_text: str) -> str:
    if not api_key:
        return "❌ Error: OpenAI API key not found. Please check your .env file."

    print("Input to GPT (gpt-3.5-turbo):", input_text)

    prompt = (
        "You are a misinformation expert. "
        "Explain in simple terms why the following content may be misinformation:\n\n"
        f"{input_text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # ✅ Changed from gpt-4 to gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250,
        )
        print("Raw OpenAI response:", response)
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Error occurred in explain_result:", str(e))
        return f"❌ Unexpected error: {str(e)}"

# Test block
if __name__ == "__main__":
    sample_text = "The COVID-19 vaccine implants microchips in people."
    explanation = explain_result(sample_text)
    print("🧠 GPT Explanation:\n", explanation)

