"""
Model evaluation: best checkpoint, TTA, clinical metrics, and reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)

if TYPE_CHECKING:
    from src.config import TrainingConfig


def load_best_model(config: TrainingConfig) -> tf.keras.Model:
    """Load the ModelCheckpoint weights saved during training."""
    path = config.best_model_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Best model not found at {path}. Train with ModelCheckpoint first."
        )
    print(f"Loading best model from: {path}")
    return tf.keras.models.load_model(path)


def predict_test_set(
    model: tf.keras.Model,
    test_generator: tf.keras.utils.Sequence,
    test_generator_tta: tf.keras.utils.Sequence | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the test set; optional TTA via flipped second pass.

    Returns:
        y_true, y_pred (class indices), y_prob (softmax probabilities)
    """
    test_generator.reset()
    y_prob = model.predict(test_generator, verbose=1)

    if test_generator_tta is not None:
        test_generator_tta.reset()
        y_prob_flip = model.predict(test_generator_tta, verbose=1)
        y_prob = (y_prob + y_prob_flip) / 2.0
        print("TTA: averaged original + horizontal-flip predictions")

    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_generator.classes
    return y_true, y_pred, y_prob


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    """Return sklearn classification report as a string."""
    return classification_report(y_true, y_pred, target_names=class_names)


def build_clinical_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    config: TrainingConfig,
    accuracy: float,
    balanced_acc: float,
    macro_f1: float,
) -> str:
    """Text summary highlighting dangerous-class recall."""
    per_class_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    recall_by_name = {
        class_names[i]: float(per_class_recall[i])
        for i in range(len(class_names))
    }

    lines = [
        "=== Clinical / imbalance-aware summary ===",
        f"Accuracy:          {accuracy:.4f}",
        f"Balanced accuracy: {balanced_acc:.4f}",
        f"Macro F1:          {macro_f1:.4f}",
        "",
        "Per-class recall:",
    ]
    for name in class_names:
        lines.append(f"  {name:6s}: {recall_by_name[name]:.4f}")

    lines.extend(["", "Priority classes (mel / akiec / bcc):"])
    for name in config.clinical_classes:
        if name in recall_by_name:
            lines.append(f"  {name:6s} recall: {recall_by_name[name]:.4f}")

    return "\n".join(lines)


def evaluate_model(
    model: tf.keras.Model,
    test_generator: tf.keras.utils.Sequence,
    class_names: list[str],
    config: TrainingConfig | None = None,
    test_generator_tta: tf.keras.utils.Sequence | None = None,
    verbose: bool = True,
) -> dict:
    """
    Full test evaluation with imbalance-aware metrics and clinical summary.
    """
    if config is None:
        from src.config import get_config

        config = get_config()

    use_tta = config.use_tta and test_generator_tta is not None
    tta_gen = test_generator_tta if use_tta else None

    y_true, y_pred, y_prob = predict_test_set(model, test_generator, tta_gen)
    report = generate_classification_report(y_true, y_pred, class_names)

    accuracy = float(np.mean(y_true == y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    clinical_summary = build_clinical_summary(
        y_true, y_pred, class_names, config, accuracy, balanced_acc, macro_f1
    )

    if verbose:
        print(report)
        print()
        print(clinical_summary)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "classification_report": report,
        "clinical_summary": clinical_summary,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
    }


def evaluate_best_checkpoint(
    test_generator: tf.keras.utils.Sequence,
    class_names: list[str],
    config: TrainingConfig,
    test_generator_tta: tf.keras.utils.Sequence | None = None,
    verbose: bool = True,
) -> dict:
    """Load best_efficientnet.h5 and run full evaluation."""
    model = load_best_model(config)
    return evaluate_model(
        model,
        test_generator,
        class_names,
        config=config,
        test_generator_tta=test_generator_tta,
        verbose=verbose,
    )


def save_evaluation_artifacts(results: dict, config: TrainingConfig) -> None:
    """Persist classification report and clinical summary."""
    from src.utils import save_text_report

    save_text_report(
        results["classification_report"], config.classification_report_path
    )
    save_text_report(results["clinical_summary"], config.clinical_summary_path)
    print(f"Report saved: {config.classification_report_path}")
    print(f"Clinical summary saved: {config.clinical_summary_path}")
