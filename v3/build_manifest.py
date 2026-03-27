"""
Build classification manifest from Roboflow dataset folders.

Scans v3/datasets for image files, maps folders to classes, and creates
a stratified train/val/test split saved as Parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import polars as pl
from sklearn.model_selection import train_test_split

# Class mapping based on folder names
# Order: 0 = report, 1 = prescription, 2 = other
CLASS_NAMES = ["report", "prescription", "other"]
FOLDER_TO_CLASS: dict[str, int] = {
    # Prescription class (label=1)
    "prescription-image-vkedm_prescription-image-labelling-xh0np_v1": 1,
    "tf-test_prescription-dpflz_v2": 1,
    # Report class (label=0)
    "medicalimage-z4t5b_prescription-oeiss_v9": 0,
    "pratiks-workspace-5pcoh_report-0jp5y_v5": 0,
    # Other class (label=2)
    "blaind_blaind_v3": 2,
    "sky-zfxvm_coco-yrx1j_v1": 2,
}

# Image extensions to scan
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Folders to exclude (relative to dataset root)
EXCLUDE_PATHS = {
    "tf-test_prescription-dpflz_v2/test/images_augmented",
}


def find_images_in_dataset(
    dataset_root: Path,
    exclude_paths: set[str],
    extensions: set[str],
) -> list[dict]:
    """
    Find all images in a dataset folder.

    Args:
        dataset_root: Path to the dataset folder
        exclude_paths: Set of relative paths to exclude
        extensions: Set of valid image extensions

    Returns:
        List of dicts with image_path, original_split info
    """
    images = []

    # Roboflow structure: train/images, valid/images, test/images
    # Or: train/, valid/, test/ directly
    # Or: images/ at root (some datasets)

    for split_dir in ["train", "valid", "test"]:
        split_path = dataset_root / split_dir
        if not split_path.exists():
            continue

        # Check for images subdirectory (Roboflow style)
        images_dir = split_path / "images"
        if images_dir.exists():
            scan_dir = images_dir
        else:
            scan_dir = split_path

        for img_file in scan_dir.iterdir():
            if img_file.suffix.lower() not in extensions:
                continue

            # Check exclusion
            rel_path = img_file.relative_to(dataset_root.parent)
            if str(rel_path.parent) in exclude_paths:
                continue

            images.append({
                "image_path": str(img_file.resolve()),
                "original_split": split_dir,
            })

    return images


def build_manifest(
    datasets_dir: Path,
    output_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Build the classification manifest.

    Args:
        datasets_dir: Path to v3/datasets
        output_path: Path for output Parquet file
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed for reproducibility

    Returns:
        Polars DataFrame with the manifest
    """
    all_records = []

    for dataset_folder, class_label in FOLDER_TO_CLASS.items():
        dataset_path = datasets_dir / dataset_folder
        if not dataset_path.exists():
            print(f"Warning: Dataset folder not found: {dataset_path}")
            continue

        images = find_images_in_dataset(
            dataset_path,
            EXCLUDE_PATHS,
            IMAGE_EXTENSIONS,
        )

        for img_info in images:
            all_records.append({
                "image_path": img_info["image_path"],
                "label": class_label,
                "source_dataset": dataset_folder,
                "original_split": img_info["original_split"],
            })

        print(f"  {dataset_folder}: {len(images)} images (class={CLASS_NAMES[class_label]})")

    print(f"\nTotal images found: {len(all_records)}")

    # Create DataFrame
    df = pl.DataFrame(all_records)

    # Print class distribution
    print("\nClass distribution before split:")
    for label, name in enumerate(CLASS_NAMES):
        count = len(df.filter(pl.col("label") == label))
        print(f"  {name}: {count}")

    # Stratified split
    # First: hold out test set
    # Then: split remaining into train/val
    labels = df["label"].to_numpy()
    indices = list(range(len(df)))

    test_frac = test_ratio / (train_ratio + val_ratio + test_ratio)
    val_frac_from_remainder = val_ratio / (train_ratio + val_ratio)

    # Split into test and rest
    rest_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        stratify=labels,
        random_state=random_state,
    )

    # Split rest into train and val
    rest_labels = [labels[i] for i in rest_idx]
    train_idx, val_idx = train_test_split(
        rest_idx,
        test_size=val_frac_from_remainder,
        stratify=rest_labels,
        random_state=random_state,
    )

    # Add split column
    split_col = [""] * len(df)
    for i in train_idx:
        split_col[i] = "train"
    for i in val_idx:
        split_col[i] = "val"
    for i in test_idx:
        split_col[i] = "test"

    df = df.with_columns(pl.Series("split", split_col))

    # Reorder columns
    df = df.select(["image_path", "label", "split", "source_dataset", "original_split"])

    # Print final split stats
    print("\nFinal split distribution:")
    for split_name in ["train", "val", "test"]:
        split_df = df.filter(pl.col("split") == split_name)
        print(f"\n{split_name.upper()} ({len(split_df)} images):")
        for label, name in enumerate(CLASS_NAMES):
            count = len(split_df.filter(pl.col("label") == label))
            print(f"  {name}: {count}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to Parquet
    df.write_parquet(output_path)
    print(f"\nManifest saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build classification manifest")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help="Path to datasets directory (default: v3/datasets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: v3/data/classification_manifest.parquet)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).resolve().parent
    datasets_dir = args.datasets_dir or script_dir / "datasets"
    output_path = args.output or script_dir / "data" / "classification_manifest.parquet"

    print(f"Datasets directory: {datasets_dir}")
    print(f"Output path: {output_path}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print()

    build_manifest(
        datasets_dir=datasets_dir,
        output_path=output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()