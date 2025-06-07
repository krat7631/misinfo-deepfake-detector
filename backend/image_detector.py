from deepface import DeepFace

def analyze_image(image_path):  # 🔁 Renamed from detect_image_deepfake
    try:
        result = DeepFace.analyze(image_path, actions=['emotion', 'age', 'gender'], enforce_detection=True)
        return {
            "real": True,
            "analysis": result
        }
    except Exception as e:
        return {
            "real": False,
            "error": str(e)
        }

