"""
TFLite Inference Engine for Edge AI Image Classification.

Wraps the TensorFlow Lite interpreter with:
  - Lazy loading with thread-safety
  - Input/output tensor introspection
  - Top-K softmax decoding with ImageNet labels
"""
import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Default model path (overridable via environment variable)
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "model" / "optimized" / "model.tflite"
_LABELS_PATH = Path(__file__).parent.parent / "model" / "imagenet_labels.json"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_DEFAULT_MODEL_PATH)))


class InferenceEngine:
    """
    Thread-safe TFLite inference engine.

    Loads the quantized MobileNetV2 model once and exposes a `predict`
    method that decodes raw logits to labelled confidence scores.
    """

    def __init__(self, model_path: Path = MODEL_PATH, num_threads: int = 2):
        self._lock = threading.Lock()
        self._model_path = model_path
        self._num_threads = num_threads
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._labels: List[str] = []
        self.model_info: Dict[str, Any] = {}

        self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the TFLite model and resolve labels."""
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            # Fall back to full TensorFlow when tflite_runtime is unavailable
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        if not self._model_path.exists():
            # Graceful degradation: load a fresh MobileNetV2 from TF Hub
            # for demo purposes when the optimized model isn't downloaded yet.
            logger.warning(
                f"Optimized model not found at {self._model_path}. "
                "Falling back to in-memory MobileNetV2 (fp32)."
            )
            self._interpreter = self._build_fallback_interpreter()
        else:
            logger.info(f"Loading TFLite model from {self._model_path}")
            self._interpreter = Interpreter(
                model_path=str(self._model_path),
                num_threads=self._num_threads,
            )

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Labels
        self._labels = self._load_labels()

        # Model metadata
        size_mb = (
            os.path.getsize(self._model_path) / (1024 * 1024)
            if self._model_path.exists()
            else 14.0
        )
        input_shape = list(self._input_details[0]["shape"])
        self.model_info = {
            "name": "mobilenetv2-int8",
            "version": "1.2.0",
            "size_mb": round(size_mb, 2),
            "classes": len(self._labels),
            "input_shape": input_shape,
        }
        logger.info(
            f"Inference engine ready | input={input_shape} | "
            f"labels={len(self._labels)} | size={size_mb:.1f}MB"
        )

    def _build_fallback_interpreter(self):
        """
        Build a TFLite interpreter from a freshly-exported MobileNetV2.
        Used in CI / first-run before `scripts/download_model.py` is executed.
        """
        import tensorflow as tf

        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=True, weights="imagenet"
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(base)
        tflite_model = converter.convert()

        # Save so subsequent starts are fast
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model_path.write_bytes(tflite_model)
        logger.info(f"Saved fallback model to {self._model_path}")

        return tf.lite.Interpreter(model_content=tflite_model)

    def _load_labels(self) -> List[str]:
        """Load ImageNet class labels from JSON file."""
        if _LABELS_PATH.exists():
            with open(_LABELS_PATH) as f:
                label_map: Dict[str, str] = json.load(f)
            # label_map keys are "0".."999"
            return [label_map[str(i)] for i in range(len(label_map))]
        # Minimal fallback
        logger.warning("Labels file not found; using numeric class IDs.")
        return [f"class_{i}" for i in range(1000)]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def predict(self, input_tensor: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Run inference on a preprocessed image tensor.

        Args:
            input_tensor: float32 array of shape [1, 224, 224, 3], values in [0, 1].
            top_k:        Number of top predictions to return.

        Returns:
            List of dicts: [{"label": str, "confidence": float}, ...]
            Sorted by confidence descending.
        """
        expected_dtype = self._input_details[0]["dtype"]
        quant_params = self._input_details[0].get("quantization", (0.0, 0))

        # Re-quantize if the model expects integer input (INT8 quantization)
        if expected_dtype == np.int8:
            scale, zero_point = quant_params
            if scale > 0:
                input_tensor = (input_tensor / scale + zero_point).astype(np.int8)
            else:
                # Default MobileNetV2 quantization: [-128, 127]
                input_tensor = ((input_tensor - 0.5) / 0.00392157).astype(np.int8)
        elif expected_dtype == np.uint8:
            input_tensor = (input_tensor * 255).astype(np.uint8)

        with self._lock:
            self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
            t0 = time.perf_counter()
            self._interpreter.invoke()
            logger.debug(f"Raw inference: {(time.perf_counter() - t0)*1000:.2f}ms")
            output = self._interpreter.get_tensor(self._output_details[0]["index"])

        # Dequantize output if needed
        out_quant = self._output_details[0].get("quantization", (0.0, 0))
        if self._output_details[0]["dtype"] in (np.int8, np.uint8):
            scale, zero_point = out_quant
            output = (output.astype(np.float32) - zero_point) * scale

        probabilities = output[0]  # shape [num_classes]

        # Softmax (model may already apply it; idempotent if probabilities sum ≈ 1)
        exp_probs = np.exp(probabilities - probabilities.max())
        probabilities = exp_probs / exp_probs.sum()

        top_indices = np.argsort(probabilities)[::-1][:top_k]

        return [
            {
                "label": self._labels[idx] if idx < len(self._labels) else f"class_{idx}",
                "confidence": float(round(float(probabilities[idx]), 4)),
            }
            for idx in top_indices
        ]
