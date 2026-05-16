"""
Central configuration for HAM10000 multiclass training.

Paths default to Kaggle Input / Working directories. Override with environment
variables when running locally (see README).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _is_kaggle() -> bool:
    return os.path.exists("/kaggle")


def _project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class TrainingConfig:
    """Hyperparameters and paths used across dataset, model, and training."""

    # --- Dataset (Kaggle Input) ---
    kaggle_dataset_slug: str = "ham10000-dataset"
    kaggle_dataset_slug_alt: str = "datasets/kmader/skin-cancer-mnist-ham10000"

    organized_dir_name: str = "ham10000_organized"

    # --- Image / batch ---
    img_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_classes: int = 7

    # --- Splits ---
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    # Group by lesion_id so the same lesion never appears in train and test.
    use_lesion_grouped_split: bool = True

    # --- Train-only minority oversampling (folder copies) ---
    oversample_minority_train: bool = True
    # Target count per minority class = min(nv_train * ratio, max_multiplier * original)
    oversample_target_ratio: float = 0.5
    oversample_max_multiplier: float = 3.0

    # --- Class weights (mel / akiec / bcc focus) ---
    class_weight_power: float = 1.0
    clinical_boost_classes: tuple[str, ...] = ("mel", "akiec", "bcc")
    clinical_boost_factor: float = 1.5

    # --- Augmentation ---
    use_strong_augmentation: bool = True
    rotation_range: int = 20
    zoom_range: float = 0.2
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    brightness_range: tuple[float, float] = (0.8, 1.2)

    # --- Model head ---
    dense_units: int = 256
    dropout_rate: float = 0.5
    label_smoothing: float = 0.1
    learning_rate: float = 3e-5
    fine_tune_learning_rate: float = 1e-5
    fine_tune_head_lr: float = 1e-4
    use_discriminative_lr: bool = True
    fine_tune_unfreeze_last_n: int = 60

    # --- Training ---
    epochs: int = 15
    fine_tune_epochs: int = 15

    # --- Callbacks ---
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.3
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"

    # --- Evaluation ---
    use_tta: bool = True
    clinical_classes: tuple[str, ...] = ("mel", "akiec", "bcc")

    # --- Output filenames ---
    best_model_filename: str = "best_efficientnet.h5"
    final_model_filename: str = "efficientnet_phase2.h5"
    history_plot_filename: str = "training_history.png"
    confusion_matrix_filename: str = "confusion_matrix.png"
    confusion_matrix_norm_filename: str = "confusion_matrix_normalized.png"
    classification_report_filename: str = "classification_report.txt"
    clinical_summary_filename: str = "clinical_summary.txt"

    # Resolved at runtime
    project_root: Path = field(default_factory=_project_root)
    kaggle_input_dir: Path | None = field(default=None, init=False)
    organized_dir: Path | None = field(default=None, init=False)
    output_dir: Path | None = field(default=None, init=False)
    models_dir: Path | None = field(default=None, init=False)
    plots_dir: Path | None = field(default=None, init=False)
    reports_dir: Path | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if _is_kaggle():
            self._setup_kaggle_paths()
        else:
            self._setup_local_paths()

    def _setup_kaggle_paths(self) -> None:
        self.kaggle_input_dir = self._resolve_kaggle_input()
        self.organized_dir = Path("/kaggle/working") / self.organized_dir_name
        self.output_dir = Path("/kaggle/working") / "outputs"
        self._ensure_output_dirs()

    def _setup_local_paths(self) -> None:
        env_input = os.environ.get("HAM10000_INPUT_DIR")
        if env_input:
            self.kaggle_input_dir = Path(env_input)
        else:
            self.kaggle_input_dir = (
                self.project_root / "data" / "ham10000-dataset"
            )
        env_organized = os.environ.get("HAM10000_ORGANIZED_DIR")
        if env_organized:
            self.organized_dir = Path(env_organized)
        else:
            self.organized_dir = (
                self.project_root / "data" / self.organized_dir_name
            )
        self.output_dir = self.project_root / "outputs"
        self._ensure_output_dirs()

    def _resolve_kaggle_input(self) -> Path:
        candidates = [
            Path("/kaggle/input") / self.kaggle_dataset_slug,
            Path("/kaggle/input") / self.kaggle_dataset_slug_alt,
            Path("/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"),
        ]
        for path in candidates:
            if path.exists():
                return path
        return Path("/kaggle/input") / self.kaggle_dataset_slug

    def _ensure_output_dirs(self) -> None:
        assert self.output_dir is not None
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        for directory in (
            self.output_dir,
            self.models_dir,
            self.plots_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def train_dir(self) -> Path:
        assert self.organized_dir is not None
        return self.organized_dir / "train"

    @property
    def val_dir(self) -> Path:
        assert self.organized_dir is not None
        return self.organized_dir / "val"

    @property
    def test_dir(self) -> Path:
        assert self.organized_dir is not None
        return self.organized_dir / "test"

    @property
    def best_model_path(self) -> Path:
        assert self.models_dir is not None
        return self.models_dir / self.best_model_filename

    @property
    def final_model_path(self) -> Path:
        assert self.models_dir is not None
        return self.models_dir / self.final_model_filename

    @property
    def history_plot_path(self) -> Path:
        assert self.plots_dir is not None
        return self.plots_dir / self.history_plot_filename

    @property
    def confusion_matrix_path(self) -> Path:
        assert self.plots_dir is not None
        return self.plots_dir / self.confusion_matrix_filename

    @property
    def confusion_matrix_norm_path(self) -> Path:
        assert self.plots_dir is not None
        return self.plots_dir / self.confusion_matrix_norm_filename

    @property
    def classification_report_path(self) -> Path:
        assert self.reports_dir is not None
        return self.reports_dir / self.classification_report_filename

    @property
    def clinical_summary_path(self) -> Path:
        assert self.reports_dir is not None
        return self.reports_dir / self.clinical_summary_filename


def get_config() -> TrainingConfig:
    """Return a fresh configuration instance (Phase 2b mel-recall defaults)."""
    return TrainingConfig()
