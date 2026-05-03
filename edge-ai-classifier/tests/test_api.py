"""
tests/test_api.py — FastAPI endpoint tests using TestClient.

Run with:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --cov=api
"""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_engine():
    """Mock InferenceEngine so tests don't need a real TFLite model."""
    engine = MagicMock()
    engine.model_info = {
        "name": "mobilenetv2-int8",
        "version": "1.2.0",
        "size_mb": 3.4,
        "classes": 1000,
        "input_shape": [1, 224, 224, 3],
    }
    engine.predict.return_value = [
        {"label": "golden retriever", "confidence": 0.921},
        {"label": "Labrador retriever", "confidence": 0.043},
        {"label": "cocker spaniel", "confidence": 0.018},
        {"label": "kuvasz", "confidence": 0.007},
        {"label": "clumber spaniel", "confidence": 0.004},
    ]
    return engine


@pytest.fixture(scope="module")
def mock_preprocessor():
    """Mock ImagePreprocessor."""
    preprocessor = MagicMock()
    preprocessor.process.return_value = np.zeros((1, 224, 224, 3), dtype=np.float32)
    return preprocessor


@pytest.fixture(scope="module")
def client(mock_engine, mock_preprocessor):
    """TestClient with mocked inference and preprocessing singletons."""
    with (
        patch("api.main.InferenceEngine", return_value=mock_engine),
        patch("api.main.ImagePreprocessor", return_value=mock_preprocessor),
    ):
        from api.main import app
        import api.main as main_module
        main_module.engine = mock_engine
        main_module.preprocessor = mock_preprocessor
        main_module.startup_time = 0.0
        yield TestClient(app)


@pytest.fixture
def jpeg_image_bytes() -> bytes:
    """Generate a minimal valid JPEG in memory."""
    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def png_image_bytes() -> bytes:
    img = Image.new("RGB", (128, 128), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class TestRoot:
    def test_root_returns_message(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "message" in resp.json()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert data["model"]["name"] == "mobilenetv2-int8"
        assert data["model"]["size_mb"] == 3.4
        assert data["model"]["classes"] == 1000

    def test_health_model_info_structure(self, client):
        resp = client.get("/api/v1/health")
        model = resp.json()["model"]
        for key in ("name", "version", "size_mb", "classes", "input_shape"):
            assert key in model, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Single classify
# ---------------------------------------------------------------------------
class TestClassify:
    def test_classify_jpeg(self, client, jpeg_image_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", jpeg_image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 5
        assert "inference_time_ms" in data
        assert "model_version" in data

    def test_classify_png(self, client, png_image_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.png", png_image_bytes, "image/png")},
        )
        assert resp.status_code == 200

    def test_classify_predictions_sorted_by_confidence(self, client, jpeg_image_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", jpeg_image_bytes, "image/jpeg")},
        )
        preds = resp.json()["predictions"]
        confidences = [p["confidence"] for p in preds]
        assert confidences == sorted(confidences, reverse=True), "Predictions not sorted by confidence"

    def test_classify_confidence_range(self, client, jpeg_image_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", jpeg_image_bytes, "image/jpeg")},
        )
        for pred in resp.json()["predictions"]:
            assert 0.0 <= pred["confidence"] <= 1.0

    def test_classify_unsupported_mime(self, client):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 415

    def test_classify_no_file(self, client):
        resp = client.post("/api/v1/classify")
        assert resp.status_code == 422  # FastAPI validation error

    def test_classify_inference_time_positive(self, client, jpeg_image_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", jpeg_image_bytes, "image/jpeg")},
        )
        assert resp.json()["inference_time_ms"] > 0


# ---------------------------------------------------------------------------
# Batch classify
# ---------------------------------------------------------------------------
class TestBatchClassify:
    def test_batch_classify_two_images(self, client, jpeg_image_bytes, png_image_bytes):
        resp = client.post(
            "/api/v1/classify/batch",
            files=[
                ("files", ("img1.jpg", jpeg_image_bytes, "image/jpeg")),
                ("files", ("img2.png", png_image_bytes, "image/png")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 2
        assert len(data["results"]) == 2

    def test_batch_classify_exceeds_limit(self, client, jpeg_image_bytes):
        files = [
            ("files", (f"img{i}.jpg", jpeg_image_bytes, "image/jpeg"))
            for i in range(17)  # MAX_BATCH_SIZE = 16
        ]
        resp = client.post("/api/v1/classify/batch", files=files)
        assert resp.status_code == 400
        assert "exceeds maximum" in resp.json()["detail"].lower()

    def test_batch_total_time_positive(self, client, jpeg_image_bytes):
        resp = client.post(
            "/api/v1/classify/batch",
            files=[("files", ("img.jpg", jpeg_image_bytes, "image/jpeg"))],
        )
        assert resp.json()["total_inference_time_ms"] > 0


# ---------------------------------------------------------------------------
# OpenAPI docs
# ---------------------------------------------------------------------------
class TestDocs:
    def test_openapi_schema_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Edge AI Image Classifier"

    def test_docs_page_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
