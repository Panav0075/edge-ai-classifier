"""
model/train.py — Fine-tune MobileNetV2 for custom image classification.

Usage:
    python model/train.py \
        --dataset  data/train \
        --val-dir  data/val \
        --epochs   15 \
        --batch    32 \
        --output   model/saved_model

The script:
  1. Loads MobileNetV2 pretrained on ImageNet (frozen base)
  2. Attaches a custom classification head
  3. Fine-tunes with learning-rate warmup + cosine decay
  4. Saves as a TF SavedModel (feeds into model/optimize.py)
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
IMG_SIZE    = (224, 224)
NUM_CLASSES = 1000          # Override for custom datasets
BASE_LR     = 1e-3
FINE_LR     = 1e-5
WARMUP_EPOCHS = 3
DROPOUT     = 0.3
LABEL_SMOOTHING = 0.1


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def build_dataset(
    data_dir: str,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline from an ImageFolder-style directory.

    Expected structure:
        data_dir/
          class_a/  *.jpg
          class_b/  *.jpg
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        label_mode="categorical",
    )

    normalization = layers.Rescaling(1.0 / 255)

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
    ]) if augment else None

    def preprocess(images, labels):
        images = normalization(images)
        if augmentation:
            images = augmentation(images, training=True)
        return images, labels

    return (
        ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int, dropout: float = DROPOUT) -> tf.keras.Model:
    """
    MobileNetV2 base + custom classification head.

    Phase 1 — train head only (base frozen).
    Phase 2 — unfreeze top 30 layers and fine-tune end-to-end.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="mobilenetv2_custom")


# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup → cosine decay."""

    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.base_lr * step / tf.cast(self.warmup_steps, tf.float32)
        cosine_arg = np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * self.base_lr * (1 + tf.math.cos(cosine_arg))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    logger.info(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    # Datasets
    train_ds = build_dataset(args.dataset, args.batch, augment=True)
    val_ds   = build_dataset(args.val_dir, args.batch, augment=False, shuffle=False)

    num_classes = len(train_ds.class_names)
    steps_per_epoch = len(train_ds)
    logger.info(f"Classes: {num_classes} | Steps/epoch: {steps_per_epoch}")

    # --------------- Phase 1: head warmup ---------------
    logger.info("=== Phase 1: Training classification head ===")
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(BASE_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )
    model.summary(line_length=100)

    cb_phase1 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        callbacks.TensorBoard(log_dir=f"{args.output}/logs/phase1"),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=cb_phase1,
    )

    # --------------- Phase 2: fine-tuning ---------------
    logger.info("=== Phase 2: Fine-tuning top layers ===")
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    total_steps = (args.epochs - WARMUP_EPOCHS) * steps_per_epoch
    lr_schedule = WarmupCosineDecay(FINE_LR, total_steps, warmup_steps=steps_per_epoch)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    cb_phase2 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            str(output_path / "best_checkpoint"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        callbacks.TensorBoard(log_dir=f"{args.output}/logs/phase2"),
        callbacks.CSVLogger(str(output_path / "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs - WARMUP_EPOCHS,
        callbacks=cb_phase2,
    )

    # Save final SavedModel
    model.save(str(output_path / "saved_model"))
    logger.info(f"Model saved to {output_path / 'saved_model'}")

    # Save class names for label mapping
    import json
    label_map = {str(i): name for i, name in enumerate(train_ds.class_names)}
    with open(output_path / "class_labels.json", "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Saved {num_classes} class labels.")

    # Final metrics
    val_metrics = model.evaluate(val_ds, verbose=0)
    logger.info(
        f"Final validation — Loss: {val_metrics[0]:.4f} | "
        f"Top-1: {val_metrics[1]:.4f} | Top-5: {val_metrics[2]:.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MobileNetV2 for edge classification")
    p.add_argument("--dataset",  required=True,            help="Path to training data directory")
    p.add_argument("--val-dir",  required=True,            help="Path to validation data directory")
    p.add_argument("--output",   default="model/saved_model", help="Output directory")
    p.add_argument("--epochs",   type=int, default=15)
    p.add_argument("--batch",    type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
