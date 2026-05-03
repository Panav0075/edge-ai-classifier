"""
scripts/download_model.py — Download the pre-optimized TFLite model.

On first run (or in CI), this exports a fresh quantized MobileNetV2 from
TensorFlow and saves it to model/optimized/model.tflite so the API can
start without running the full optimize.py pipeline.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --quantize fp16
    python scripts/download_model.py --force   # re-export even if file exists
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path(__file__).parent.parent / "model" / "optimized"
LABELS_PATH = Path(__file__).parent.parent / "model" / "imagenet_labels.json"


def export_mobilenetv2(quantize: str = "int8") -> Path:
    """Export MobileNetV2 pretrained weights as a TFLite model."""
    import tensorflow as tf
    import numpy as np

    logger.info(f"Loading MobileNetV2 (imagenet weights)...")
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=True, weights="imagenet"
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize == "fp16":
        converter.target_spec.supported_types = [tf.float16]
        logger.info("Quantization: FP16")
    elif quantize == "int8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

        def representative_dataset():
            for _ in range(100):
                yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

        converter.representative_dataset = representative_dataset
        logger.info("Quantization: INT8 (synthetic calibration)")
    else:
        logger.info("Quantization: FP32 (none)")

    logger.info("Converting to TFLite (this may take a minute)...")
    tflite_bytes = converter.convert()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "model.tflite"
    out_path.write_bytes(tflite_bytes)

    size_mb = len(tflite_bytes) / (1024 * 1024)
    logger.info(f"Saved {size_mb:.1f} MB model → {out_path}")
    return out_path


def write_imagenet_labels() -> None:
    """Write a minimal ImageNet label map if not present."""
    if LABELS_PATH.exists():
        return
    try:
        import tensorflow as tf
        # Decode using MobileNetV2's built-in decoder for 1 dummy image
        import numpy as np
        dummy = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.zeros((1, 224, 224, 3), dtype=np.float32)
        )
        model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
        preds = model.predict(dummy, verbose=0)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1000)[0]
        label_map = {str(i): name for i, (_, name, _) in enumerate(decoded)}
        LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LABELS_PATH, "w") as f:
            json.dump(label_map, f, indent=2)
        logger.info(f"Saved ImageNet labels → {LABELS_PATH}")
    except Exception as exc:
        logger.warning(f"Could not write labels: {exc}. Falling back to numeric IDs.")
        label_map = {str(i): f"class_{i}" for i in range(1000)}
        with open(LABELS_PATH, "w") as f:
            json.dump(label_map, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download / export the optimized TFLite model")
    p.add_argument("--quantize", default="int8", choices=["fp32", "fp16", "int8"])
    p.add_argument("--force",    action="store_true", help="Re-export even if model already exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = OUTPUT_DIR / "model.tflite"

    if model_path.exists() and not args.force:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model already exists ({size_mb:.1f} MB) at {model_path}. Use --force to re-export.")
    else:
        export_mobilenetv2(quantize=args.quantize)

    write_imagenet_labels()
    logger.info("✅ Model ready. You can now run: uvicorn api.main:app --reload")
