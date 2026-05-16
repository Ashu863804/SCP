"""
Training loop, callbacks, and fine-tuning.

Phase A changes:
  A1 — compile_fine_tune_model no longer attempts list-based Adam LR (Keras doesn't
       support per-variable learning rates via Adam(learning_rate=list[float])).
       The broken try/except fallback is removed; a clean single-LR compile is used.
  A2 — fine_tune_label_smoothing (0.0) is used during fine-tuning so minority-class
       gradients are not diluted in the final training phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

if TYPE_CHECKING:
    from src.config import TrainingConfig


def get_callbacks(config: TrainingConfig) -> list[tf.keras.callbacks.Callback]:
    """EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint on val_loss."""
    return [
        EarlyStopping(
            monitor=config.checkpoint_monitor,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            mode=config.checkpoint_mode,
        ),
        ReduceLROnPlateau(
            monitor=config.checkpoint_monitor,
            patience=config.reduce_lr_patience,
            factor=config.reduce_lr_factor,
            mode=config.checkpoint_mode,
        ),
        ModelCheckpoint(
            str(config.best_model_path),
            monitor=config.checkpoint_monitor,
            mode=config.checkpoint_mode,
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_model(
    model: tf.keras.Model,
    train_generator: tf.keras.utils.Sequence,
    val_generator: tf.keras.utils.Sequence,
    config: TrainingConfig,
    class_weight_dict: dict[int, float] | None = None,
    callbacks: list[tf.keras.callbacks.Callback] | None = None,
) -> Any:
    """Fit the frozen-backbone model."""
    if callbacks is None:
        callbacks = get_callbacks(config)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )
    return history


def compile_fine_tune_model(
    model: tf.keras.Model,
    config: TrainingConfig,
) -> tf.keras.Model:
    """
    Compile the model for fine-tuning.

    A1: Uses a single Adam LR (fine_tune_learning_rate) for the whole model.
        Per-variable LR via Adam(learning_rate=list) is not supported by
        tf.keras — attempting it silently falls back to a scalar and was
        therefore removed.
    A2: Uses fine_tune_label_smoothing (default 0.0) so minority-class
        gradient signal is not diluted during the critical fine-tune phase.
    """
    from src.model import compile_model

    compile_model(
        model,
        learning_rate=config.fine_tune_learning_rate,
        label_smoothing=config.fine_tune_label_smoothing,
    )
    print(
        f"Fine-tune compile: LR={config.fine_tune_learning_rate}, "
        f"label_smoothing={config.fine_tune_label_smoothing}"
    )
    return model


def fine_tune_model(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_generator: tf.keras.utils.Sequence,
    val_generator: tf.keras.utils.Sequence,
    config: TrainingConfig,
    class_weight_dict: dict[int, float] | None = None,
    callbacks: list[tf.keras.callbacks.Callback] | None = None,
) -> Any:
    """Unfreeze top backbone layers and fine-tune with lower learning rate(s)."""
    base_model.trainable = True
    for layer in base_model.layers[: -config.fine_tune_unfreeze_last_n]:
        layer.trainable = False

    compile_fine_tune_model(model, config)

    if callbacks is None:
        callbacks = get_callbacks(config)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.fine_tune_epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )
    return history


def save_final_model(model: tf.keras.Model, path: str | Any) -> None:
    """Persist a Keras model to disk."""
    path = str(path)
    model.save(path)
    print(f"Model saved to: {path}")


def save_final_from_best(config: TrainingConfig) -> tf.keras.Model:
    """Copy best checkpoint to final_model_path for deployment."""
    from src.evaluate import load_best_model

    model = load_best_model(config)
    save_final_model(model, config.final_model_path)
    return model
