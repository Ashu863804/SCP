"""
EfficientNetB0 transfer-learning model (frozen backbone + custom head).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model

if TYPE_CHECKING:
    from src.config import TrainingConfig


def build_efficientnet_model(
    num_classes: int,
    config: TrainingConfig | None = None,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> tuple[Model, tf.keras.Model]:
    """
    Build EfficientNetB0 with frozen ImageNet weights and a classification head.

    Architecture (same as original notebook):
        EfficientNetB0 (frozen) -> GAP -> BatchNorm -> Dense(256) ->
        Dropout(0.5) -> Softmax(num_classes)
    """
    if config is None:
        from src.config import get_config

        config = get_config()

    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(config.dense_units, activation="relu")(x)
    x = Dropout(config.dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def compile_model(
    model: Model,
    learning_rate: float,
    label_smoothing: float = 0.1,
) -> Model:
    """Compile with Adam, label-smoothed categorical cross-entropy, accuracy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=["accuracy"],
    )
    return model


def create_and_compile_model(
    num_classes: int,
    config: TrainingConfig | None = None,
) -> tuple[Model, tf.keras.Model]:
    """Build and compile the Phase 2 model in one step."""
    if config is None:
        from src.config import get_config

        config = get_config()

    model, base_model = build_efficientnet_model(num_classes, config=config)
    compile_model(
        model,
        learning_rate=config.learning_rate,
        label_smoothing=config.label_smoothing,
    )
    return model, base_model
