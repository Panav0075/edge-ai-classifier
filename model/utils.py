"""
model/utils.py — Shared utilities for training, optimization, and evaluation.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_imagenet_labels(labels_path: str | Path) -> List[str]:
    """Load ImageNet class labels from a JSON file keyed by string index."""
    with open(labels_path) as f:
        mapping: Dict[str, str] = json.load(f)
    return [mapping[str(i)] for i in range(len(mapping))]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def top_k_predictions(
    logits: np.ndarray,
    labels: List[str],
    k: int = 5,
) -> List[Dict[str, float | str]]:
    """Return top-k {label, confidence} dicts from raw logits."""
    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:k]
    return [
        {"label": labels[i], "confidence": float(round(float(probs[i]), 4))}
        for i in top_indices
    ]


def get_model_size_mb(path: str | Path) -> float:
    """Compute total size of a model directory or single file in MB."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)


def print_model_summary(model_info: Dict) -> None:
    """Pretty-print model metadata."""
    print(
        f"\n{'═'*50}\n"
        f"  Model        : {model_info.get('name', 'unknown')}\n"
        f"  Version      : {model_info.get('version', '—')}\n"
        f"  Size         : {model_info.get('size_mb', '—')} MB\n"
        f"  Classes      : {model_info.get('classes', '—')}\n"
        f"  Input shape  : {model_info.get('input_shape', '—')}\n"
        f"{'═'*50}\n"
    )


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy and TensorFlow."""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
