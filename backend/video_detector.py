FRAME_SAMPLE_INTERVAL = 30
SUSPICIOUS_THRESHOLD = 2


def analyze_video(file_path):
    if not file_path:
        raise ValueError("Video path is required.")

    import cv2
    from deepface import DeepFace

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video file.")

    frame_count, deepfake_score = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % FRAME_SAMPLE_INTERVAL == 0:
            try:
                DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
            except ValueError:
                deepfake_score += 1
            except RuntimeError:
                deepfake_score += 1

    cap.release()

    return {
        "total_frames": frame_count,
        "suspicious_frames": deepfake_score,
        "is_deepfake": deepfake_score > SUSPICIOUS_THRESHOLD
    }
