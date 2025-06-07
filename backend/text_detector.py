from transformers import pipeline

# ✅ Publicly available fake news detection model
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

def analyze_text(text: str) -> dict:
    result = classifier(text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 3)
    }




