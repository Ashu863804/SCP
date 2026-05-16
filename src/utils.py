"""
Shared helpers: plotting, saving artifacts, and notebook path setup.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def add_src_to_path() -> Path:
    """
    Add the project ``src`` package to ``sys.path``.

    Use at the top of Kaggle notebooks after cloning the GitHub repo into
    ``/kaggle/working/skin-cancer-detection``.
    """
    here = Path(__file__).resolve().parent.parent
    candidates = [
        Path("/kaggle/working/skin-cancer-detection"),
        Path.cwd(),
        Path.cwd().parent,
        here,
    ]
    seen: set[str] = set()
    for root in candidates:
        root = root.resolve()
        src = root / "src"
        if (src / "config.py").exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            if root_str not in seen:
                seen.add(root_str)
            return root
    raise FileNotFoundError(
        "Could not locate project src/. Clone the repo into /kaggle/working "
        "or run the notebook from the repository root."
    )


def plot_training_history(
    history: Any,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot accuracy and loss curves for train and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
    show: bool = True,
    title: str = "Confusion Matrix — EfficientNet",
) -> np.ndarray:
    """Draw and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return cm


def plot_sample_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    class_names: list[str],
    n_samples: int = 12,
    show: bool = True,
) -> None:
    """Visualize a batch of test images with true vs predicted labels."""
    n_samples = min(n_samples, len(images))
    cols = 4
    rows = int(np.ceil(n_samples / cols))

    plt.figure(figsize=(15, 8))
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        display_img = images[i]
        if display_img.max() > 1.0:
            display_img = np.clip(display_img, 0, 255).astype(np.uint8)
        plt.imshow(display_img)

        true_name = class_names[int(np.argmax(true_labels[i]))]
        pred_name = class_names[int(np.argmax(pred_probs[i]))]
        color = "green" if true_name == pred_name else "red"
        plt.title(f"T:{true_name}\nP:{pred_name}", color=color, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def save_text_report(content: str, path: Path) -> None:
    """Write a string report (e.g. classification report) to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
