"""
Model evaluation: predictions, classification report, and test metrics.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


def predict_test_set(
    model: tf.keras.Model,
    test_generator: tf.keras.utils.Sequence,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the full test set (generator must have shuffle=False).

    Returns:
        y_true: integer class indices
        y_pred: integer predicted indices
    """
    test_generator.reset()
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    return y_true, y_pred


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    """Return sklearn classification report as a string."""
    return classification_report(
        y_true, y_pred, target_names=class_names
    )


def evaluate_model(
    model: tf.keras.Model,
    test_generator: tf.keras.utils.Sequence,
    class_names: list[str],
    verbose: bool = True,
) -> dict:
    """
    Full test evaluation: predictions + classification report.

    Returns a dict with y_true, y_pred, and report text.
    """
    y_true, y_pred = predict_test_set(model, test_generator)
    report = generate_classification_report(y_true, y_pred, class_names)

    if verbose:
        print(report)

    accuracy = float(np.mean(y_true == y_pred))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "classification_report": report,
        "accuracy": accuracy,
    }
