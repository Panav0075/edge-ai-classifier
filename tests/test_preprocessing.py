"""
tests/test_preprocessing.py — Unit tests for the image preprocessing pipeline.

Tests each pipeline stage independently, ensuring correctness and robustness
against edge-case inputs (tiny images, RGBA, grayscale, corrupt data).

Run with: pytest tests/test_preprocessing.py -v
"""
import io
import struct

import numpy as np
import pytest
from PIL import Image

from api.preprocessing import ImagePreprocessor, extract_features_batch, TARGET_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_image_bytes(
    width: int = 256,
    height: int = 256,
    mode: str = "RGB",
    fmt: str = "JPEG",
    color=None,
) -> bytes:
    color = color or (100, 150, 200)
    if mode == "RGBA":
        color = (*color, 180)
    elif mode == "L":
        color = 128
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    save_fmt = "PNG" if mode in ("RGBA", "LA", "L") else fmt
    img.save(buf, format=save_fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ImagePreprocessor
# ---------------------------------------------------------------------------
class TestImagePreprocessorDecode:
    def test_decode_valid_jpeg(self):
        preprocessor = ImagePreprocessor()
        img = preprocessor._decode(make_image_bytes())
        assert isinstance(img, Image.Image)

    def test_decode_valid_png(self):
        preprocessor = ImagePreprocessor()
        img = preprocessor._decode(make_image_bytes(fmt="PNG"))
        assert isinstance(img, Image.Image)

    def test_decode_invalid_bytes_raises(self):
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="Could not decode image"):
            preprocessor._decode(b"this is not an image at all")

    def test_decode_empty_bytes_raises(self):
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError):
            preprocessor._decode(b"")


class TestImagePreprocessorToRGB:
    preprocessor = ImagePreprocessor()

    def test_rgb_unchanged(self):
        img = Image.new("RGB", (10, 10))
        result = self.preprocessor._to_rgb(img)
        assert result.mode == "RGB"
        assert result is img  # same object — no conversion

    def test_rgba_to_rgb(self):
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        result = self.preprocessor._to_rgb(img)
        assert result.mode == "RGB"

    def test_grayscale_to_rgb(self):
        img = Image.new("L", (10, 10), 128)
        result = self.preprocessor._to_rgb(img)
        assert result.mode == "RGB"

    def test_palette_to_rgb(self):
        img = Image.new("P", (10, 10))
        result = self.preprocessor._to_rgb(img)
        assert result.mode == "RGB"


class TestCenterCropResize:
    preprocessor = ImagePreprocessor()

    def test_output_size_exact(self):
        img = Image.new("RGB", (500, 300))
        result = self.preprocessor._center_crop_resize(img, TARGET_SIZE)
        assert result.size == (TARGET_SIZE[1], TARGET_SIZE[0])  # PIL uses (W, H)

    def test_square_input(self):
        img = Image.new("RGB", (300, 300))
        result = self.preprocessor._center_crop_resize(img, TARGET_SIZE)
        assert result.size == (TARGET_SIZE[1], TARGET_SIZE[0])

    def test_tiny_image(self):
        img = Image.new("RGB", (10, 10))
        result = self.preprocessor._center_crop_resize(img, TARGET_SIZE)
        assert result.size == (TARGET_SIZE[1], TARGET_SIZE[0])

    def test_landscape_input(self):
        img = Image.new("RGB", (800, 300))
        result = self.preprocessor._center_crop_resize(img, TARGET_SIZE)
        assert result.size == (TARGET_SIZE[1], TARGET_SIZE[0])

    def test_portrait_input(self):
        img = Image.new("RGB", (300, 800))
        result = self.preprocessor._center_crop_resize(img, TARGET_SIZE)
        assert result.size == (TARGET_SIZE[1], TARGET_SIZE[0])


class TestNormalization:
    preprocessor = ImagePreprocessor()

    def test_normalize_range(self):
        arr = np.array([[[0, 128, 255]] * 5] * 5, dtype=np.float32)
        normalized = self.preprocessor._normalize(arr)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_zero_maps_to_zero(self):
        arr = np.zeros((5, 5, 3), dtype=np.float32)
        assert np.allclose(self.preprocessor._normalize(arr), 0.0)

    def test_normalize_255_maps_to_one(self):
        arr = np.full((5, 5, 3), 255.0, dtype=np.float32)
        assert np.allclose(self.preprocessor._normalize(arr), 1.0)

    def test_standardize_changes_range(self):
        arr = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        standardized = self.preprocessor._standardize(arr)
        # After standardization values may leave [0,1]
        assert not np.allclose(standardized, arr)


class TestFullPipeline:
    preprocessor = ImagePreprocessor()

    def test_output_shape(self):
        tensor = self.preprocessor.process(make_image_bytes())
        assert tensor.shape == (1, 224, 224, 3)

    def test_output_dtype_float32(self):
        tensor = self.preprocessor.process(make_image_bytes())
        assert tensor.dtype == np.float32

    def test_output_values_in_range(self):
        tensor = self.preprocessor.process(make_image_bytes())
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_rgba_input(self):
        tensor = self.preprocessor.process(make_image_bytes(mode="RGBA"))
        assert tensor.shape == (1, 224, 224, 3)

    def test_grayscale_input(self):
        tensor = self.preprocessor.process(make_image_bytes(mode="L"))
        assert tensor.shape == (1, 224, 224, 3)

    def test_non_square_input(self):
        tensor = self.preprocessor.process(make_image_bytes(width=800, height=400))
        assert tensor.shape == (1, 224, 224, 3)

    def test_corrupt_bytes_raises_value_error(self):
        with pytest.raises(ValueError):
            self.preprocessor.process(b"\xff\xd8\xff garbage data !@#$")


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------
class TestExtractFeaturesBatch:
    def test_batch_shape(self):
        images = [make_image_bytes() for _ in range(4)]
        batch = extract_features_batch(images)
        assert batch.shape == (4, 224, 224, 3)

    def test_single_image_batch(self):
        images = [make_image_bytes()]
        batch = extract_features_batch(images)
        assert batch.shape == (1, 224, 224, 3)

    def test_batch_values_in_range(self):
        images = [make_image_bytes() for _ in range(3)]
        batch = extract_features_batch(images)
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0
