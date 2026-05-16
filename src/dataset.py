"""
HAM10000 dataset loading, stratified splits, and Keras image generators.

Mirrors the original Kaggle notebook:
- Read metadata CSV from Kaggle Input
- Map images from part_1 / part_2 folders
- 80/20 train-test, then 10% validation from train
- Copy into class folders for ``flow_from_directory``
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
    image_path_dict = _build_image_path_index(input_dir)
    df["path"] = df["image_id"].map(image_path_dict)
    df = df[df["path"].notna()].reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
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
            image_id = filename.stem
            image_path_dict[image_id] = str(filename)

    print(f"Total images found: {len(image_path_dict)}")
    return image_path_dict


def organize_train_val_test(
    df: pd.DataFrame,
    config: TrainingConfig,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """
    Create ``train/``, ``val/``, and ``test/`` folder trees with class subdirs.

    Same split logic as the original notebook (stratified, random_state=42).
    """
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
            return train_dir, val_dir, test_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)

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

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    train_dir = config.train_dir
    val_dir = config.val_dir
    test_dir = config.test_dir

    print("\nOrganizing TRAIN...")
    _copy_split(train_df, train_dir)
    print("Organizing VALIDATION...")
    _copy_split(val_df, val_dir)
    print("Organizing TEST...")
    _copy_split(test_df, test_dir)
    print("\nDone!")
    return train_dir, val_dir, test_dir


def _copy_split(dataframe: pd.DataFrame, split_dir: Path) -> None:
    """Copy images into ``split_dir/<class_name>/``."""
    for _, row in dataframe.iterrows():
        class_folder = split_dir / row["dx"]
        class_folder.mkdir(parents=True, exist_ok=True)
        dest = class_folder / Path(row["path"]).name
        if not dest.exists():
            shutil.copy(row["path"], dest)


def create_data_generators(
    config: TrainingConfig,
) -> tuple[
    ImageDataGenerator,
    ImageDataGenerator,
    ImageDataGenerator,
    tf.keras.utils.Sequence,
    tf.keras.utils.Sequence,
    tf.keras.utils.Sequence,
]:
    """
    Build train/val/test ``ImageDataGenerator`` instances and directory flows.

    Augmentation matches the original notebook (EfficientNet preprocess + aug).
    """
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    target_size = config.img_size
    batch_size = config.batch_size

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )
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
    return (
        train_datagen,
        val_datagen,
        test_datagen,
        train_generator,
        val_generator,
        test_generator,
    )


def compute_smoothed_class_weights(
    train_generator: tf.keras.utils.Sequence,
    power: float = 0.5,
) -> dict[int, float]:
    """
    Balanced class weights with sqrt smoothing (same as original notebook).
    """
    labels = train_generator.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    class_weights = np.power(class_weights, power)
    return dict(enumerate(class_weights))


def get_class_names(train_generator: tf.keras.utils.Sequence) -> list[str]:
    """Return human-readable class names in index order."""
    return list(train_generator.class_indices.keys())


def prepare_dataset_pipeline(
    config: TrainingConfig,
    force_organize: bool = False,
) -> dict:
    """
    End-to-end dataset setup: metadata -> folders -> generators -> weights.

    Returns a dict consumed by the training notebook.
    """
    assert config.kaggle_input_dir is not None
    df = load_ham10000_metadata(config.kaggle_input_dir)
    organize_train_val_test(df, config, force=force_organize)
    (
        _train_datagen,
        _val_datagen,
        _test_datagen,
        train_gen,
        val_gen,
        test_gen,
    ) = create_data_generators(config)
    class_weight_dict = compute_smoothed_class_weights(
        train_gen, power=config.class_weight_power
    )
    class_names = get_class_names(train_gen)

    return {
        "dataframe": df,
        "train_generator": train_gen,
        "val_generator": val_gen,
        "test_generator": test_gen,
        "class_weight_dict": class_weight_dict,
        "class_names": class_names,
        "num_classes": train_gen.num_classes,
    }
