"""
MobileNetV4 3-class fine-tuning training script.

Fine-tunes timm's mobilenetv4_conv_small on three classes:
- 0: report
- 1: prescription
- 2: other

Features:
- Polars-backed Parquet manifest
- Albumentations augmentation pipeline
- Benchmark-driven epoch cap for ~7 hour wall-clock budget
- Early stopping on validation loss
- Metrics: accuracy, per-class precision/recall, confusion matrix
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import timm
from timm.data import resolve_model_data_config, create_transform

# Import augmentation pipeline
from augmentations import MEDICAL_DOC_AUGMENT

# Class names (must match manifest builder)
CLASS_NAMES = ["report", "prescription", "other"]


class ClassificationDataset(Dataset):
    """
    Image classification dataset with albumentations support.
    """

    def __init__(
        self,
        manifest_df: pl.DataFrame,
        split: Literal["train", "val", "test"],
        transform: A.Compose | None = None,
        timm_transform=None,
    ):
        """
        Args:
            manifest_df: Polars DataFrame with image_path, label, split columns
            split: Which split to use
            transform: Albumentations transform (for training augmentation)
            timm_transform: timm transform for normalization
        """
        self.split_df = manifest_df.filter(pl.col("split") == split)
        self.image_paths = self.split_df["image_path"].to_list()
        self.labels = self.split_df["label"].to_list()
        self.transform = transform
        self.timm_transform = timm_transform

        print(f"  {split}: {len(self.image_paths)} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image (BGR)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Convert BGR to RGB for albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply albumentations (if provided)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle grayscale output from augmentation
        if len(image.shape) == 2:
            # Single channel, convert to 3 channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # Convert to float32 and CHW format
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Convert to tensor
        image = torch.from_numpy(image)

        # Apply timm normalization
        if self.timm_transform is not None:
            # timm transform expects tensor input
            image = self.timm_transform(image)

        return image, label


def create_val_transform() -> A.Compose:
    """Create validation transform (no augmentation, just grayscale conversion)."""
    return A.Compose([
        A.ToGray(p=1.0),
    ])


def compute_class_weights(manifest_df: pl.DataFrame, split: str = "train") -> torch.Tensor:
    """Compute class weights based on training set distribution."""
    train_df = manifest_df.filter(pl.col("split") == split)
    label_counts = train_df.group_by("label").len().sort("label")

    counts = []
    for label in range(len(CLASS_NAMES)):
        row = label_counts.filter(pl.col("label") == label)
        if len(row) > 0:
            counts.append(row["len"][0])
        else:
            counts.append(1)

    counts = torch.tensor(counts, dtype=torch.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return weights


def run_benchmark(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_batches: int = 100,
) -> tuple[float, float]:
    """
    Run a benchmark to estimate seconds per epoch.

    Returns:
        Tuple of (train_seconds_per_batch, val_seconds_per_batch)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Benchmark training batches
    train_times = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Benchmarking training on {num_batches} batches...")
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        start = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_times.append(time.perf_counter() - start)

    # Benchmark validation batches
    model.eval()
    val_times = []
    print(f"Benchmarking validation on {min(num_batches, len(val_loader))} batches...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i >= num_batches:
                break

            images = images.to(device)

            start = time.perf_counter()
            _ = model(images)
            val_times.append(time.perf_counter() - start)

    median_train = np.median(train_times)
    median_val = np.median(val_times)

    print(f"  Median train batch time: {median_train:.4f}s")
    print(f"  Median val batch time: {median_val:.4f}s")

    return median_train, median_val


def compute_max_epochs(
    target_seconds: float,
    n_train_batches: int,
    n_val_batches: int,
    train_batch_time: float,
    val_batch_time: float,
    safety_factor: float = 0.92,
    min_epochs: int = 2,
) -> int:
    """
    Compute max epochs to fit within target wall-clock time.
    """
    seconds_per_epoch = (
        n_train_batches * train_batch_time +
        n_val_batches * val_batch_time
    )

    effective_budget = target_seconds * safety_factor
    max_epochs = int(effective_budget / seconds_per_epoch)
    max_epochs = max(min_epochs, max_epochs)

    estimated_time = max_epochs * seconds_per_epoch

    print(f"\nTime budget calculation:")
    print(f"  Target wall-clock: {target_seconds/3600:.1f} hours")
    print(f"  Seconds per epoch: {seconds_per_epoch:.1f}s")
    print(f"  Max epochs (with {safety_factor} safety): {max_epochs}")
    print(f"  Estimated total time: {estimated_time/3600:.1f} hours")

    return max_epochs


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=total_loss/total, acc=100.*correct/total)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate the model.

    Returns:
        Tuple of (average loss, accuracy, all_labels, all_predictions)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [val]")
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

            pbar.set_postfix(loss=total_loss/total, acc=100.*correct/total)

    avg_loss = total_loss / total
    accuracy = correct / total
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return avg_loss, accuracy, all_labels, all_preds


def compute_per_class_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute per-class precision, recall, and F1."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )

    metrics = {}
    for i, name in enumerate(class_names):
        metrics[name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
        }
    return metrics


from safetensors.torch import save_file


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    output_path: Path,
    class_names: list[str],
):
    """Save model checkpoint in safetensors format."""
    # Save model weights in safetensors format
    weights_path = output_path.with_suffix(".safetensors")
    save_file(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    # Save metadata in JSON
    metadata = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "class_names": class_names,
        "label_map": {name: i for i, name in enumerate(class_names)},
    }
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    # Also save full checkpoint for resuming training
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "class_names": class_names,
        "label_map": {name: i for i, name in enumerate(class_names)},
    }
    torch.save(checkpoint, output_path)
    print(f"Full checkpoint saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MobileNetV4 3-class fine-tuning")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest Parquet file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--target-hours",
        type=float,
        default=7.0,
        help="Target wall-clock time in hours",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights in loss function",
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Determine paths
    script_dir = Path(__file__).resolve().parent
    manifest_path = args.manifest or script_dir / "data" / "classification_manifest.parquet"
    output_dir = args.output_dir or script_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {manifest_path}")
    print(f"Output directory: {output_dir}")

    # Load manifest
    print("\nLoading manifest...")
    manifest_df = pl.read_parquet(manifest_path)
    print(f"Total samples: {len(manifest_df)}")

    # Create model
    print("\nCreating model...")
    model_name = "mobilenetv4_conv_small.e3600_r256_in1k"
    model = timm.create_model(model_name, pretrained=True, num_classes=3)
    model = model.to(device)

    # Get timm data config for normalization
    data_config = resolve_model_data_config(model)
    timm_transform = create_transform(**data_config, is_training=False)
    print(f"Model: {model_name}")
    print(f"Data config: {data_config}")

    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(manifest_df, split="train")
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ClassificationDataset(
        manifest_df,
        split="train",
        transform=MEDICAL_DOC_AUGMENT,
        timm_transform=timm_transform,
    )
    val_dataset = ClassificationDataset(
        manifest_df,
        split="val",
        transform=create_val_transform(),
        timm_transform=timm_transform,
    )
    test_dataset = ClassificationDataset(
        manifest_df,
        split="test",
        transform=create_val_transform(),
        timm_transform=timm_transform,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"\nDataloader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Run benchmark to estimate epoch time
    print("\n" + "="*60)
    print("Running preflight benchmark...")
    print("="*60)
    train_batch_time, val_batch_time = run_benchmark(
        model, train_loader, val_loader, device, num_batches=50
    )

    # Compute max epochs
    target_seconds = args.target_hours * 3600
    max_epochs = compute_max_epochs(
        target_seconds=target_seconds,
        n_train_batches=len(train_loader),
        n_val_batches=len(val_loader),
        train_batch_time=train_batch_time,
        val_batch_time=val_batch_time,
        safety_factor=0.92,
        min_epochs=2,
    )

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_val_loss = float("inf")
    patience_counter = 0
    train_history = []
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, val_labels, val_preds = validate(
            model, val_loader, criterion, device, epoch
        )

        # Scheduler step
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Record history
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time,
        })

        # Compute per-class metrics
        class_metrics = compute_per_class_metrics(val_labels, val_preds, CLASS_NAMES)

        # Print epoch summary
        print(f"\nEpoch {epoch} summary:")
        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        print(f"  Epoch time: {epoch_time:.1f}s")
        print(f"  Per-class metrics:")
        for name, m in class_metrics.items():
            print(f"    {name}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                output_dir / "best.pt", CLASS_NAMES
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        # Time check
        elapsed = time.time() - start_time
        if elapsed > target_seconds * 0.95:
            print(f"\nTime budget reached ({elapsed/3600:.1f}h > {args.target_hours*0.95:.1f}h)")
            break

    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Final test evaluation
    print("\nEvaluating on test set...")
    best_checkpoint = torch.load(output_dir / "best.pt", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_acc, test_labels, test_preds = validate(
        model, test_loader, criterion, device, epoch=0
    )
    test_metrics = compute_per_class_metrics(test_labels, test_preds, CLASS_NAMES)

    print(f"\nTest set results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Per-class metrics:")
    for name, m in test_metrics.items():
        print(f"    {name}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")

    # Save test results - convert numpy types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    test_results = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "per_class_metrics": convert_to_native(test_metrics),
        "total_training_time_hours": float(total_time / 3600),
        "epochs_trained": len(train_history),
    }
    test_results_path = output_dir / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Test results saved to {test_results_path}")


if __name__ == "__main__":
    main()