"""
Image Preprocessing Pipeline for Edge AI Classification.

Implements a configurable, efficient pipeline that mirrors the exact
preprocessing used during MobileNetV2 training:
  1. Decode raw bytes → PIL Image
  2. Convert to RGB
  3. Resize with aspect-ratio-preserving center crop → 224×224
  4. Normalize to [0, 1] (float32)
  5. Expand dims → [1, H, W, C] batch tensor
"""
import io
import logging
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# MobileNetV2 expected input
TARGET_SIZE: Tuple[int, int] = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet mean per channel
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # ImageNet std  per channel


class ImagePreprocessor:
    """
    Stateless image preprocessing pipeline.

    All methods are pure functions (no internal state mutation) so the
    single shared instance is safe to call from multiple threads.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = TARGET_SIZE,
        normalize: bool = True,
        standardize: bool = False,
    ):
        """
        Args:
            target_size:   (H, W) the model expects.
            normalize:     Divide pixel values by 255 → [0, 1].
            standardize:   Apply ImageNet channel-wise mean/std normalization
                           on top of normalization. Disabled by default since
                           the TFLite model's input layer handles this internally.
        """
        self.target_size = target_size
        self.normalize = normalize
        self.standardize = standardize

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def process(self, image_bytes: bytes) -> np.ndarray:
        """
        Full preprocessing pipeline: bytes → model-ready tensor.

        Args:
            image_bytes: Raw image file content (JPEG / PNG / WebP / BMP).

        Returns:
            float32 ndarray of shape [1, H, W, 3] ready for TFLite inference.

        Raises:
            ValueError: If the bytes cannot be decoded as an image.
        """
        image = self._decode(image_bytes)
        image = self._to_rgb(image)
        image = self._center_crop_resize(image, self.target_size)
        array = self._to_array(image)
        array = self._normalize(array)
        if self.standardize:
            array = self._standardize(array)
        tensor = self._add_batch_dim(array)
        logger.debug(f"Preprocessed tensor: shape={tensor.shape} dtype={tensor.dtype}")
        return tensor

    def process_file(self, path: str) -> np.ndarray:
        """Convenience wrapper: read from disk path and process."""
        with open(path, "rb") as f:
            return self.process(f.read())

    # ------------------------------------------------------------------
    # Pipeline stages (each is a pure, independently-testable function)
    # ------------------------------------------------------------------
    @staticmethod
    def _decode(image_bytes: bytes) -> Image.Image:
        """Decode raw bytes into a PIL Image."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # Force decompression to catch corrupt files early
            return image
        except Exception as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        """
        Ensure the image is in RGB mode.
        Handles: RGBA (transparency → white bg), grayscale L/LA, CMYK, P.
        """
        if image.mode == "RGB":
            return image
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        return image.convert("RGB")

    @staticmethod
    def _center_crop_resize(image: Image.Image, target: Tuple[int, int]) -> Image.Image:
        """
        Resize shortest edge to target size, then center-crop.

        This matches the standard ImageNet eval preprocessing (no distortion).
        """
        th, tw = target
        w, h = image.size

        # Resize so the shorter side equals the target
        ratio = max(tw / w, th / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Center crop
        left   = (new_w - tw) // 2
        top    = (new_h - th) // 2
        right  = left + tw
        bottom = top  + th
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _to_array(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to float32 NumPy array of shape [H, W, 3]."""
        return np.array(image, dtype=np.float32)

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        """Scale pixel values from [0, 255] to [0.0, 1.0]."""
        return array / 255.0

    @staticmethod
    def _standardize(array: np.ndarray) -> np.ndarray:
        """Apply ImageNet channel-wise mean/std normalisation."""
        return (array - MEAN) / STD

    @staticmethod
    def _add_batch_dim(array: np.ndarray) -> np.ndarray:
        """Expand [H, W, C] → [1, H, W, C] to match TFLite input shape."""
        return np.expand_dims(array, axis=0)


# ---------------------------------------------------------------------------
# Feature extraction helpers (used by model/train.py)
# ---------------------------------------------------------------------------
def extract_features_batch(
    images: list,
    preprocessor: ImagePreprocessor | None = None,
) -> np.ndarray:
    """
    Preprocess a list of PIL Images or byte strings into a batched tensor.

    Args:
        images:       List of raw bytes or PIL.Image objects.
        preprocessor: Optional shared ImagePreprocessor instance.

    Returns:
        float32 ndarray of shape [N, H, W, 3].
    """
    if preprocessor is None:
        preprocessor = ImagePreprocessor()

    tensors = []
    for img in images:
        if isinstance(img, bytes):
            t = preprocessor.process(img)  # [1, H, W, 3]
        else:
            # PIL Image
            arr = preprocessor._to_array(preprocessor._center_crop_resize(img, TARGET_SIZE))
            arr = preprocessor._normalize(arr)
            t = preprocessor._add_batch_dim(arr)
        tensors.append(t)

    return np.concatenate(tensors, axis=0)  # [N, H, W, 3]
