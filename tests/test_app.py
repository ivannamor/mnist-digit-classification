"""Tests for the web demo (app.py)."""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

# Demo dependencies live in requirements-demo.txt, not requirements.txt:
# without them this whole file is skipped instead of failing at import.
gr = pytest.importorskip("gradio")
gradio_client = pytest.importorskip("gradio_client")

import app  # noqa: E402

needs_checkpoint = pytest.mark.skipif(not app.CHECKPOINT.exists(), reason="committed CNN checkpoint not present")


def _canvas_drawing_of_a_one() -> Image.Image:
    """What the Sketchpad produces: RGBA, transparent background, black stroke."""
    image = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
    ImageDraw.Draw(image).line([(140, 40), (140, 240)], fill=(0, 0, 0, 255), width=18)
    return image


# --- classify() -------------------------------------------------------------


@needs_checkpoint
def test_classify_returns_probability_per_digit():
    result = app.classify({"composite": _canvas_drawing_of_a_one()})
    assert set(result) == {str(d) for d in range(10)}
    assert sum(result.values()) == pytest.approx(1.0)
    assert max(result, key=result.get) == "1"


@pytest.mark.parametrize("drawing", [None, {"composite": None}, {"composite": Image.new("RGBA", (280, 280))}])
def test_classify_empty_canvas_returns_none(drawing):
    assert app.classify(drawing) is None


# --- The running app, through Gradio's own client --------------------------


@needs_checkpoint
def test_demo_end_to_end(tmp_path):
    """Sends a drawing the way the browser does, so Gradio's own payload
    conversion (e.g. flattening the canvas onto white) is exercised too."""
    path = tmp_path / "one.png"
    _canvas_drawing_of_a_one().save(path)

    demo = app.build_demo()
    _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        client = gradio_client.Client(url, verbose=False)
        payload = {"background": None, "layers": [], "composite": gradio_client.handle_file(str(path))}
        result = client.predict(payload, api_name="/classify")
    finally:
        demo.close()

    assert result["label"] == "1"
    assert len(result["confidences"]) == 3
