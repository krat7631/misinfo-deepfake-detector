from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_text_success(monkeypatch):
    monkeypatch.setattr(main_module, "analyze_text", lambda _: {"label": "fake", "score": 0.98})
    monkeypatch.setattr(main_module, "explain_result", lambda _: "Potential misinformation cues found.")

    response = client.post("/analyze-text", data={"text": "Aliens landed yesterday"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["label"] == "fake"
    assert "explanation" in payload


def test_reject_empty_text():
    response = client.post("/analyze-text", data={"text": "   "})
    assert response.status_code == 400


def test_reject_unsupported_image_extension():
    file_bytes = b"dummy"
    response = client.post(
        "/analyze-image",
        files={"file": ("payload.exe", file_bytes, "application/octet-stream")},
    )
    assert response.status_code == 400


def test_reject_unsupported_video_extension():
    file_bytes = b"dummy"
    response = client.post(
        "/analyze-video",
        files={"file": ("payload.exe", file_bytes, "application/octet-stream")},
    )
    assert response.status_code == 400
