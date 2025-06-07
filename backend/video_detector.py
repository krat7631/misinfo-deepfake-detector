import cv2
from deepface import DeepFace

def analyze_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_count, deepfake_score = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Analyze one frame per second (assuming 30 fps)
        if frame_count % 30 == 0:
            try:
                DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
            except:
                deepfake_score += 1

    cap.release()

    return {
        "total_frames": frame_count,
        "suspicious_frames": deepfake_score,
        "is_deepfake": deepfake_score > 2  # You can tweak this threshold
    }
