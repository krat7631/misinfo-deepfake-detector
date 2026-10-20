import numpy as np
import pytest

from backend.face_crop import FaceBox, crop_face, largest_face_crop
from backend.text_preprocessing import build_model_input, normalize_article
from backend.video_detector import summarize_frame_scores


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("WASHINGTON (Reuters) - The Senate voted on Tuesday.", "The Senate voted on Tuesday."),
        ("Big news today. Featured image via Getty Images", "Big news today."),
        ("Watch this https://t.co/abc pic.twitter.com/xyz now", "Watch this now"),
        ("Thanks @someone for the tip", "Thanks for the tip"),
    ],
)
def test_normalize_article_strips_publisher_fingerprints(raw, expected):
    assert normalize_article(raw) == expected


def test_build_model_input_joins_title_and_body():
    assert build_model_input("Title", "Body text.") == "Title Body text."
    assert build_model_input(None, "  Body only ") == "Body only"


def test_crop_face_stays_inside_frame_without_padding():
    rgb = np.zeros((100, 120, 3), dtype=np.uint8)
    rgb[:, :60] = 255  # left half white, so a padded crop would show black or white streaks
    crop = crop_face(rgb, FaceBox(x=0, y=0, width=50, height=50, score=0.9), size=32)
    assert crop.size == (32, 32)


def test_crop_face_shrinks_when_face_box_exceeds_image():
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    crop = crop_face(rgb, FaceBox(x=0, y=0, width=40, height=40, score=0.9), size=16)
    assert crop.size == (16, 16)


def test_blank_image_has_no_face():
    from PIL import Image

    assert largest_face_crop(Image.new("RGB", (200, 200), "gray")) is None


def test_select_faces_skips_small_and_edge_clipped_background_faces():
    from backend.image_detector import select_faces

    main = FaceBox(x=60, y=40, width=70, height=95, score=0.93)
    clipped = FaceBox(x=12, y=1, width=50, height=50, score=0.86)  # touches the top edge
    tiny = FaceBox(x=150, y=150, width=30, height=30, score=0.95)
    unsure = FaceBox(x=150, y=60, width=60, height=60, score=0.65)
    second = FaceBox(x=140, y=150, width=55, height=60, score=0.9)

    selected, skipped = select_faces([main, clipped, second, unsure, tiny], width=200, height=300)

    assert selected == [main, second]
    assert skipped == 3


def test_select_faces_always_keeps_the_largest_face():
    from backend.image_detector import select_faces

    only = FaceBox(x=0, y=0, width=30, height=30, score=0.7)
    assert select_faces([only], width=100, height=100) == ([only], 0)
    assert select_faces([], width=100, height=100) == ([], 0)


def test_manipulated_segment_in_real_video_is_flagged():
    # Real for 3.5 s, then a swapped face for 3 s: the whole-clip mean is below 0.5.
    scores = [0.03] * 7 + [0.99] * 6
    timestamps = [i * 0.5 for i in range(13)]
    summary = summarize_frame_scores(scores, timestamps)

    assert summary["mean_frame_probability"] < 0.5
    assert summary["label"] == "fake"
    assert summary["fake_probability"] == 0.99
    assert summary["most_suspicious_segment"] == {"start": 3.5, "end": 4.5}
    assert summary["share_of_frames_flagged"] == round(6 / 13, 4)


def test_single_noisy_frame_does_not_flip_verdict():
    summary = summarize_frame_scores([0.05, 0.05, 0.95, 0.05, 0.05])
    assert summary["label"] == "real"
    assert summary["max_frame_probability"] == 0.95


def test_short_video_uses_all_frames():
    summary = summarize_frame_scores([0.8, 0.6])
    assert summary["fake_probability"] == 0.7


def test_summarize_frame_scores_requires_faces():
    from backend.image_detector import NoFaceError

    with pytest.raises(NoFaceError):
        summarize_frame_scores([])
