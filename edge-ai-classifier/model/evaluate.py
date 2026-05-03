"""
model/evaluate.py — Evaluate TFLite model accuracy and performance.

Computes Top-1 / Top-5 accuracy on a validation set and generates
a confusion matrix + per-class breakdown report.

Usage:
    python model/evaluate.py \
        --model  model/optimized/model.tflite \
        --data   data/val \
        --output reports/eval_report.json
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# TFLite runner
# ---------------------------------------------------------------------------
class TFLiteEvaluator:
    def __init__(self, model_path: str, num_threads: int = 4):
        try:
            import tflite_runtime.interpreter as tflite
            self._interpreter = tflite.Interpreter(
                model_path=model_path, num_threads=num_threads
            )
        except ImportError:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(
                model_path=model_path, num_threads=num_threads
            )

        self._interpreter.allocate_tensors()
        self._input  = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]

    def predict_single(self, image_array: np.ndarray) -> np.ndarray:
        """Run inference on a single preprocessed image tensor [1,H,W,3]."""
        inp = image_array.astype(self._input["dtype"])
        if self._input["dtype"] == np.int8:
            scale, zp = self._input["quantization"]
            inp = (inp / (scale or 1.0) + zp).astype(np.int8)
        self._interpreter.set_tensor(self._input["index"], inp)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output["index"]).copy()
        if self._output["dtype"] in (np.int8, np.uint8):
            scale, zp = self._output["quantization"]
            output = (output.astype(np.float32) - zp) * (scale or 1.0)
        return output[0]  # shape [num_classes]


# ---------------------------------------------------------------------------
# Preprocessing (inline, no dependency on api/)
# ---------------------------------------------------------------------------
def preprocess(path: Path, img_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    ratio = max(img_size[0] / h, img_size[1] / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - img_size[1]) // 2
    top  = (new_h - img_size[0]) // 2
    img  = img.crop((left, top, left + img_size[1], top + img_size[0]))
    arr  = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(
    model_path: str,
    data_dir: str,
    label_map: Dict[str, str] | None = None,
    max_samples: int | None = None,
) -> Dict:
    evaluator = TFLiteEvaluator(model_path)
    data_path = Path(data_dir)

    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    logger.info(f"Found {len(classes)} classes in {data_dir}")

    total = correct_top1 = correct_top5 = 0
    per_class_correct: Dict[str, int] = {c: 0 for c in classes}
    per_class_total:   Dict[str, int] = {c: 0 for c in classes}
    latencies: List[float] = []

    image_paths = []
    for cls in classes:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            image_paths.extend((data_path / cls).glob(ext))

    if max_samples:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:max_samples]

    logger.info(f"Evaluating on {len(image_paths)} images...")

    for img_path in image_paths:
        cls_name = img_path.parent.name
        true_idx = class_to_idx[cls_name]

        try:
            tensor = preprocess(img_path)
        except Exception as exc:
            logger.warning(f"Skipping {img_path}: {exc}")
            continue

        t0 = time.perf_counter()
        logits = evaluator.predict_single(tensor)
        latencies.append((time.perf_counter() - t0) * 1000)

        top5 = np.argsort(logits)[::-1][:5]
        if top5[0] == true_idx:
            correct_top1 += 1
            per_class_correct[cls_name] += 1
        if true_idx in top5:
            correct_top5 += 1

        per_class_total[cls_name] += 1
        total += 1

        if total % 500 == 0:
            logger.info(
                f"  Progress {total}/{len(image_paths)} | "
                f"Top-1: {correct_top1/total:.3f} | Top-5: {correct_top5/total:.3f}"
            )

    results = {
        "total_samples":    total,
        "top1_accuracy":    round(correct_top1 / total, 4) if total else 0,
        "top5_accuracy":    round(correct_top5 / total, 4) if total else 0,
        "avg_latency_ms":   round(float(np.mean(latencies)), 2),
        "p95_latency_ms":   round(float(np.percentile(latencies, 95)), 2),
        "per_class_accuracy": {
            cls: round(per_class_correct[cls] / per_class_total[cls], 4)
            for cls in classes if per_class_total[cls] > 0
        },
    }

    logger.info(
        f"\n{'─'*50}\n"
        f"  Results on {total} samples\n"
        f"  Top-1 Accuracy : {results['top1_accuracy']:.2%}\n"
        f"  Top-5 Accuracy : {results['top5_accuracy']:.2%}\n"
        f"  Avg Latency    : {results['avg_latency_ms']} ms\n"
        f"  P95 Latency    : {results['p95_latency_ms']} ms\n"
        f"{'─'*50}"
    )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TFLite model accuracy and latency")
    p.add_argument("--model",       required=True,  help="Path to .tflite model")
    p.add_argument("--data",        required=True,  help="Validation data directory (ImageFolder structure)")
    p.add_argument("--output",      default="reports/eval_report.json")
    p.add_argument("--max-samples", type=int, default=None, help="Cap number of evaluation images")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate(args.model, args.data, max_samples=args.max_samples)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report saved to {out_path}")
