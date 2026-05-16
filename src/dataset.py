"""
HAM10000 dataset loading, lesion-grouped splits, oversampling, and generators.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if TYPE_CHECKING:
    from src.config import TrainingConfig


def load_ham10000_metadata(input_dir: Path) -> pd.DataFrame:
    """Load HAM10000 metadata and attach filesystem paths to each image."""
    csv_path = input_dir / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {csv_path}. Attach the HAM10000 Kaggle dataset."
        )

    df = pd.read_csv(csv_path)
    if "lesion_id" not in df.columns:
        raise ValueError(
            "HAM10000_metadata.csv must contain 'lesion_id' for grouped splits."
        )

    image_path_dict = _build_image_path_index(input_dir)
    df["path"] = df["image_id"].map(image_path_dict)
    df = df[df["path"].notna()].reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Unique lesions: {df['lesion_id'].nunique()}")
    print("\nClass distribution:")
    print(df["dx"].value_counts())
    print("\nAll class names:", df["dx"].unique().tolist())
    print(f"Images matched with metadata: {len(df)}")
    return df


def _build_image_path_index(input_dir: Path) -> dict[str, str]:
    """Index image_id -> absolute path across HAM10000 part folders."""
    image_path_dict: dict[str, str] = {}
    search_folders = [
        input_dir / "HAM10000_images_part_1",
        input_dir / "HAM10000_images_part_2",
        input_dir / "ham10000_images_part_1",
        input_dir / "ham10000_images_part_2",
    ]

    for folder in search_folders:
        if not folder.exists():
            continue
        for filename in folder.iterdir():
            if filename.suffix.lower() not in (".jpg", ".png"):
                continue
            image_path_dict[filename.stem] = str(filename)

    print(f"Total images found: {len(image_path_dict)}")
    return image_path_dict


def _split_train_val_test(
    df: pd.DataFrame, config: TrainingConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split with optional lesion grouping (no leakage across splits)."""
    if config.use_lesion_grouped_split:
        groups = df["lesion_id"]
        gss_test = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        train_idx, test_idx = next(gss_test.split(df, groups=groups))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=config.val_size,
            random_state=config.random_state,
        )
        train_groups = train_df["lesion_id"]
        sub_train_idx, val_idx = next(
            gss_val.split(train_df, groups=train_groups)
        )
        sub_train_df = train_df.iloc[sub_train_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        train_df = sub_train_df
        print("Split method: lesion-grouped (GroupShuffleSplit)")
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=df["dx"],
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=config.val_size,
            random_state=config.random_state,
            stratify=train_df["dx"],
        )
        print("Split method: stratified per-image")

    return train_df, val_df, test_df


def organize_train_val_test(
    df: pd.DataFrame,
    config: TrainingConfig,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Create train/val/test folder trees; train may include oversampled copies."""
    output_dir = config.organized_dir
    assert output_dir is not None

    if output_dir.exists() and not force:
        train_dir, val_dir, test_dir = (
            config.train_dir,
            config.val_dir,
            config.test_dir,
        )
        if train_dir.exists() and val_dir.exists() and test_dir.exists():
            print(f"Using existing organized data at: {output_dir}")
            print(
                "Tip: set force_organize=True after changing split/oversample settings."
            )
            return train_dir, val_dir, test_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_df, val_df, test_df = _split_train_val_test(df, config)

    print(f"Train size: {len(train_df)} images")
    print(f"Validation size: {len(val_df)} images")
    print(f"Test size: {len(test_df)} images")

    train_dir = config.train_dir
    val_dir = config.val_dir
    test_dir = config.test_dir

    print("\nOrganizing TRAIN...")
    _copy_split(train_df, train_dir, oversample=config.oversample_minority_train, config=config)
    print("Organizing VALIDATION...")
    _copy_split(val_df, val_dir, oversample=False, config=config)
    print("Organizing TEST...")
    _copy_split(test_df, test_dir, oversample=False, config=config)
    print("\nDone!")
    return train_dir, val_dir, test_dir


def _copy_one_row(row: pd.Series, class_folder: Path, dest_name: str | None = None) -> None:
    """Copy a single image row into class_folder."""
    class_folder.mkdir(parents=True, exist_ok=True)
    src = Path(row["path"])
    dest = class_folder / (dest_name or src.name)
    if not dest.exists():
        shutil.copy(row["path"], dest)


def _copy_split(
    dataframe: pd.DataFrame,
    split_dir: Path,
    oversample: bool,
    config: TrainingConfig,
) -> None:
    """Copy images into split_dir/<class>/; optionally oversample minority train classes."""
    for _, row in dataframe.iterrows():
        _copy_one_row(row, split_dir / row["dx"])

    if not oversample:
        return

    counts = dataframe["dx"].value_counts()
    nv_count = int(counts.get("nv", counts.max()))
    target = int(nv_count * config.oversample_target_ratio)
    print(f"  Oversampling minority train classes toward ~{target} images each")

    for class_name, count in counts.items():
        if class_name == "nv":
            continue
        max_allowed = int(count * config.oversample_max_multiplier)
        goal = min(target, max_allowed)
        if goal <= count:
            continue

        class_rows = dataframe[dataframe["dx"] == class_name].reset_index(drop=True)
        extra_needed = goal - count
        class_folder = split_dir / class_name
        for i in range(extra_needed):
            row = class_rows.iloc[i % len(class_rows)]
            stem = Path(row["path"]).stem
            suffix = Path(row["path"]).suffix
            dest_name = f"{stem}_os{i}{suffix}"
            _copy_one_row(row, class_folder, dest_name=dest_name)
        print(f"    {class_name}: {count} -> {goal} (+{extra_needed} copies)")


def _build_train_datagen(config: TrainingConfig) -> ImageDataGenerator:
    """Training generator with base or stronger augmentation."""
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    kwargs: dict = {
        "preprocessing_function": preprocess,
        "rotation_range": config.rotation_range,
        "zoom_range": config.zoom_range,
        "horizontal_flip": True,
        "fill_mode": "nearest",
    }
    if config.use_strong_augmentation:
        kwargs.update(
            width_shift_range=config.width_shift_range,
            height_shift_range=config.height_shift_range,
            brightness_range=config.brightness_range,
        )
    return ImageDataGenerator(**kwargs)


def create_data_generators(
    config: TrainingConfig,
) -> tuple[
    ImageDataGenerator,
    ImageDataGenerator,
    ImageDataGenerator,
    ImageDataGenerator | None,
    tf.keras.utils.Sequence,
    tf.keras.utils.Sequence,
    tf.keras.utils.Sequence,
    tf.keras.utils.Sequence | None,
]:
    """Build train/val/test flows; optional TTA test flow (horizontal flip)."""
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    target_size = config.img_size
    batch_size = config.batch_size

    train_datagen = _build_train_datagen(config)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
        str(config.train_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )
    val_generator = val_datagen.flow_from_directory(
        str(config.val_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )
    test_generator = test_datagen.flow_from_directory(
        str(config.test_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_generator_tta = None
    test_datagen_tta = None
    if config.use_tta:

        def preprocess_flip(image: np.ndarray) -> np.ndarray:
            return preprocess(np.fliplr(image))

        test_datagen_tta = ImageDataGenerator(preprocessing_function=preprocess_flip)
        test_generator_tta = test_datagen_tta.flow_from_directory(
            str(config.test_dir),
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    return (
        train_datagen,
        val_datagen,
        test_datagen,
        test_datagen_tta,
        train_generator,
        val_generator,
        test_generator,
        test_generator_tta,
    )


def compute_clinical_class_weights(
    train_generator: tf.keras.utils.Sequence,
    config: TrainingConfig,
) -> tuple[dict[int, float], dict[str, float]]:
    """
    Balanced class weights with power smoothing + boost for clinical classes.
    """
    class_indices = train_generator.class_indices
    labels = train_generator.classes

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    smoothed = np.power(raw_weights, config.class_weight_power)
    weight_dict = {int(i): float(w) for i, w in enumerate(smoothed)}

    boosted_display: dict[str, float] = {}
    for class_name in config.clinical_boost_classes:
        if class_name not in class_indices:
            continue
        idx = int(class_indices[class_name])
        weight_dict[idx] *= config.clinical_boost_factor
        boosted_display[class_name] = weight_dict[idx]

    ordered = {name: weight_dict[class_indices[name]] for name in sorted(class_indices)}
    print("\nClass weights (index order):")
    for name, w in ordered.items():
        mark = " *" if name in config.clinical_boost_classes else ""
        print(f"  {name}: {w:.4f}{mark}")

    return weight_dict, boosted_display


def get_class_names(train_generator: tf.keras.utils.Sequence) -> list[str]:
    """Return class names sorted by Keras class index."""
    indices = train_generator.class_indices
    return [name for name, _ in sorted(indices.items(), key=lambda x: x[1])]


def prepare_dataset_pipeline(
    config: TrainingConfig,
    force_organize: bool = False,
) -> dict:
    """End-to-end dataset setup for the training notebook."""
    assert config.kaggle_input_dir is not None
    df = load_ham10000_metadata(config.kaggle_input_dir)
    organize_train_val_test(df, config, force=force_organize)

    (
        _train_datagen,
        _val_datagen,
        _test_datagen,
        _test_datagen_tta,
        train_gen,
        val_gen,
        test_gen,
        test_gen_tta,
    ) = create_data_generators(config)

    class_weight_dict, clinical_weights = compute_clinical_class_weights(
        train_gen, config
    )
    class_names = get_class_names(train_gen)

    return {
        "dataframe": df,
        "train_generator": train_gen,
        "val_generator": val_gen,
        "test_generator": test_gen,
        "test_generator_tta": test_gen_tta,
        "class_weight_dict": class_weight_dict,
        "clinical_weights": clinical_weights,
        "class_names": class_names,
        "num_classes": train_gen.num_classes,
    }
