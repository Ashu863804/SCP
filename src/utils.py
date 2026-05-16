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


def find_project_root() -> Path:
    """
    Find the repository root (folder that contains ``src/config.py``).

    Checks common Kaggle layouts: cloned repo, flat working copy, or
    read-only copy under ``/kaggle/input``.
    """
    here = Path(__file__).resolve().parent.parent
    candidates: list[Path] = [
        Path("/kaggle/working/SCP"),
        Path("/kaggle/working/SCP/skin-cancer-detection"),
        Path("/kaggle/working"),
        here,
        Path.cwd(),
        Path.cwd().parent,
    ]

    # Parent folders of the current notebook working directory
    candidates.extend(list(Path.cwd().parents)[:6])

    # Datasets attached under /kaggle/input (e.g. GitHub repo as Kaggle dataset)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for child in sorted(kaggle_input.iterdir()):
            candidates.append(child)

    checked: set[Path] = set()
    for root in candidates:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in checked:
            continue
        checked.add(root)
        if (root / "src" / "config.py").is_file():
            return root

    raise FileNotFoundError(
        "Could not find src/config.py.\n"
        "On Kaggle, run in a prior cell:\n"
        "  !git clone https://github.com/Ashu863804/SCP.git /kaggle/working/SCP\n"
        "Then restart the kernel and run this notebook again."
    )


def add_src_to_path(chdir: bool = True) -> Path:
    """
    Add the project root to ``sys.path`` so ``import src...`` works.

    Args:
        chdir: If True, change the process cwd to the project root (recommended
            on Kaggle so outputs land in a predictable place).
    """
    root = find_project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if chdir:
        import os

        os.chdir(root)
    return root


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
    normalize: bool = False,
) -> np.ndarray:
    """Draw and optionally save a confusion matrix heatmap (counts or normalized)."""
    cm = confusion_matrix(y_true, y_pred)
    display = cm.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(
            display,
            row_sums,
            out=np.zeros_like(display),
            where=row_sums != 0,
        )
        fmt = ".2f"
        plot_title = title + " (normalized)"
    else:
        fmt = "d"
        plot_title = title

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(plot_title)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return cm


def merge_histories(*histories: Any) -> Any:
    """Concatenate Keras History objects for plotting multi-phase training."""
    if not histories:
        raise ValueError("At least one history is required")

    merged: dict[str, list] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)

    class _MergedHistory:
        history = merged

    return _MergedHistory()


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
