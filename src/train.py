"""
Training loop, callbacks, and optional fine-tuning (top layers only).
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
    """
    Callbacks matching the original notebook:
    - EarlyStopping (patience=5, restore_best_weights)
    - ReduceLROnPlateau (patience=3, factor=0.3)
    - ModelCheckpoint (best validation weights)
    """
    return [
        EarlyStopping(patience=config.early_stopping_patience, restore_best_weights=True),
        ReduceLROnPlateau(
            patience=config.reduce_lr_patience,
            factor=config.reduce_lr_factor,
        ),
        ModelCheckpoint(
            str(config.best_model_path),
            save_best_only=True,
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
    """
    Fit the frozen-backbone model (default 15 epochs with class weights).
    """
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


def fine_tune_model(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_generator: tf.keras.utils.Sequence,
    val_generator: tf.keras.utils.Sequence,
    config: TrainingConfig,
    class_weight_dict: dict[int, float] | None = None,
    callbacks: list[tf.keras.callbacks.Callback] | None = None,
) -> Any:
    """
    Unfreeze only the last N layers of EfficientNetB0 and train with lower LR.

    Same strategy as the original notebook (last 40 layers, Adam 1e-5).
    """
    base_model.trainable = True
    for layer in base_model.layers[: -config.fine_tune_unfreeze_last_n]:
        layer.trainable = False

    from src.model import compile_model

    compile_model(
        model,
        learning_rate=config.fine_tune_learning_rate,
        label_smoothing=config.label_smoothing,
    )

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
    """Persist the trained Keras model."""
    path = str(path)
    model.save(path)
    print(f"Model saved to: {path}")
