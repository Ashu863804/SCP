"""
Training loop, callbacks, and fine-tuning with optional discriminative learning rates.
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
    Compile for fine-tuning with one LR or per-variable discriminative LRs.
    """
    from src.model import compile_model

    if not config.use_discriminative_lr:
        return compile_model(
            model,
            learning_rate=config.fine_tune_learning_rate,
            label_smoothing=config.label_smoothing,
        )

    learning_rates: list[float] = []
    for var in model.trainable_variables:
        name = var.name.lower()
        if "efficientnet" in name:
            learning_rates.append(config.fine_tune_learning_rate)
        else:
            learning_rates.append(config.fine_tune_head_lr)

    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=config.label_smoothing
            ),
            metrics=["accuracy"],
        )
        print(
            "Fine-tune compile: discriminative LR "
            f"(backbone={config.fine_tune_learning_rate}, "
            f"head={config.fine_tune_head_lr})"
        )
    except (TypeError, ValueError) as exc:
        print(f"Discriminative LR not supported ({exc}); using single LR.")
        compile_model(
            model,
            learning_rate=config.fine_tune_learning_rate,
            label_smoothing=config.label_smoothing,
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
