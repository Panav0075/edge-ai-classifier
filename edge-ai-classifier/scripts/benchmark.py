"""
scripts/benchmark.py — CLI tool for benchmarking the TFLite model on a local image.

Usage:
    # Single image
    python scripts/benchmark.py --image path/to/photo.jpg

    # Stress test (100 runs for latency stats)
    python scripts/benchmark.py --image path/to/photo.jpg --runs 100

    # Point at a custom model
    python scripts/benchmark.py --image photo.jpg --model model/optimized/model.tflite
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.preprocessing import ImagePreprocessor
from api.inference import InferenceEngine


def benchmark(args: argparse.Namespace) -> None:
    # Override model path if provided
    if args.model:
        import os
        os.environ["MODEL_PATH"] = args.model

    print(f"\n{'─'*55}")
    print("  🔬 Edge AI Classifier — Benchmark")
    print(f"{'─'*55}")

    print(f"  Loading model...")
    engine      = InferenceEngine()
    preprocessor = ImagePreprocessor()

    info = engine.model_info
    print(f"  Model   : {info['name']} v{info['version']}")
    print(f"  Size    : {info['size_mb']} MB")
    print(f"  Classes : {info['classes']}")
    print(f"  Input   : {info['input_shape']}")
    print(f"{'─'*55}")

    print(f"  Image   : {args.image}")
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    # Warm-up
    tensor = preprocessor.process(image_bytes)
    for _ in range(3):
        engine.predict(tensor, top_k=5)

    # Timed runs
    latencies = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        predictions = engine.predict(tensor, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

    print(f"\n  📊 Latency ({args.runs} runs, CPU):")
    print(f"     Avg : {np.mean(latencies):.2f} ms")
    print(f"     P50 : {np.percentile(latencies, 50):.2f} ms")
    print(f"     P95 : {np.percentile(latencies, 95):.2f} ms")
    print(f"     P99 : {np.percentile(latencies, 99):.2f} ms")
    print(f"     Min : {np.min(latencies):.2f} ms")
    print(f"     Max : {np.max(latencies):.2f} ms")

    print(f"\n  🏆 Top-{len(predictions)} Predictions:")
    for rank, pred in enumerate(predictions, 1):
        bar = "█" * int(pred["confidence"] * 30)
        print(f"     {rank}. {pred['label']:<30} {pred['confidence']:.2%}  {bar}")

    print(f"{'─'*55}\n")

    if args.output:
        result = {
            "image": args.image,
            "model": info,
            "predictions": predictions,
            "latency_stats": {
                "avg_ms":  round(float(np.mean(latencies)), 2),
                "p50_ms":  round(float(np.percentile(latencies, 50)), 2),
                "p95_ms":  round(float(np.percentile(latencies, 95)), 2),
                "p99_ms":  round(float(np.percentile(latencies, 99)), 2),
            },
        }
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"  Results saved to {args.output}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark TFLite image classifier")
    p.add_argument("--image",  required=True,  help="Path to test image")
    p.add_argument("--model",  default=None,   help="Override model path")
    p.add_argument("--runs",   type=int, default=20, help="Number of inference runs")
    p.add_argument("--output", default=None,   help="Save JSON results to path")
    return p.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
