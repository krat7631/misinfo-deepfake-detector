from fastapi.testclient import TestClient

import backend.main as main_module
from backend.image_detector import NoFaceError
from backend.main import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_text_success(monkeypatch):
    monkeypatch.setattr(main_module, "analyze_text", lambda _: {"label": "fake", "fake_probability": 0.98})
    monkeypatch.setattr(main_module, "explain_text", lambda _: {"tokens": [], "top_fake_signals": []})
    monkeypatch.setattr(main_module, "llm_explanations_enabled", lambda: False)

    response = client.post("/analyze-text", data={"text": "Aliens landed yesterday"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["label"] == "fake"
    assert "attribution" in payload
    assert "explanation" not in payload


def test_analyze_text_without_attribution(monkeypatch):
    monkeypatch.setattr(main_module, "analyze_text", lambda _: {"label": "real", "fake_probability": 0.1})
    monkeypatch.setattr(main_module, "llm_explanations_enabled", lambda: False)

    response = client.post("/analyze-text", data={"text": "Rain expected on Tuesday", "skip_explanation": "true"})

    assert response.status_code == 200
    assert "attribution" not in response.json()


def test_reject_empty_text():
    response = client.post("/analyze-text", data={"text": "   "})
    assert response.status_code == 400


def test_reject_oversized_text():
    response = client.post("/analyze-text", data={"text": "a" * (main_module.MAX_TEXT_CHARS + 1)})
    assert response.status_code == 413


def test_reject_unsupported_image_extension():
    response = client.post(
        "/analyze-image",
        files={"file": ("payload.exe", b"dummy", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_reject_unsupported_video_extension():
    response = client.post(
        "/analyze-video",
        files={"file": ("payload.exe", b"dummy", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_image_without_face_returns_422(monkeypatch):
    def no_face(_):
        raise NoFaceError("No face detected.")

    monkeypatch.setattr(main_module, "analyze_image", no_face)
    response = client.post("/analyze-image", files={"file": ("photo.jpg", b"jpeg", "image/jpeg")})
    assert response.status_code == 422


def test_reject_upload_over_size_limit(monkeypatch):
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/analyze-image", files={"file": ("photo.jpg", b"x" * 11, "image/jpeg")})
    assert response.status_code == 413


def test_uploaded_file_is_deleted_after_analysis(monkeypatch):
    seen = {}

    def fake_analyze(path):
        seen["path"] = path
        return {"label": "real"}

    monkeypatch.setattr(main_module, "analyze_image", fake_analyze)
    response = client.post("/analyze-image", files={"file": ("photo.png", b"png", "image/png")})

    assert response.status_code == 200
    assert not main_module.Path(seen["path"]).exists()
