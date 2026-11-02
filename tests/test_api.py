from io import BytesIO

import anyio
import httpx
from PIL import Image

from boardwatch import api


class FakeDetector:
    def predict(self, image, score_threshold=0.55):
        assert image.size == (32, 24)
        assert score_threshold == 0.55
        return [{"label": "white_circle_cell", "score": 0.91, "box": [2.0, 3.0, 12.0, 14.0]}]


def jpeg_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), (240, 240, 240)).save(output, format="JPEG")
    return output.getvalue()


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def make_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    return anyio.run(make_request)


def test_health_and_demo_page_are_available():
    assert request("GET", "/health").status_code == 200
    response = request("GET", "/")
    assert response.status_code == 200
    assert "BoardWatch" in response.text


def test_detect_returns_class_and_box(monkeypatch):
    monkeypatch.setattr(api, "get_detector", lambda: FakeDetector())
    response = request("POST", "/detect", files={"file": ("board.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["score_threshold"] == 0.55
    assert data["detections"][0]["label"] == "white_circle_cell"
    assert data["detections"][0]["box"] == [2.0, 3.0, 12.0, 14.0]


def test_detect_rejects_non_image_upload():
    response = request("POST", "/detect", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400


def test_detect_rejects_invalid_threshold():
    response = request(
        "POST",
        "/detect?score_threshold=1.5",
        files={"file": ("board.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 422
