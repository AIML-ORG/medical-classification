"""
Minimal inference class for MobileNetV4 3-class classification.

Usage:
    python v3/infer.py <path> [--checkpoint v3/outputs/v1.pt]

    <path> can be:
    - Single image file
    - Folder with images (results.txt saved in that folder)
    - Root folder (recursively finds all folders with images, saves results.txt in each)

    Results are saved to results.txt in each folder containing images.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config, create_transform
from safetensors.torch import load_file


class ImageClassifier:
    """Minimal classifier for MobileNetV4 3-class model."""

    CLASS_NAMES = ["report", "prescription", "other"]
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, checkpoint_path: Union[str, Path], device: str = "auto"):
        """
        Initialize the classifier.

        Args:
            checkpoint_path: Path to checkpoint file (.safetensors or .pt)
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create model
        self.model = timm.create_model(
            "mobilenetv4_conv_small.e3600_r256_in1k",
            pretrained=False,
            num_classes=3
        )

        # Load weights based on file format
        if self.checkpoint_path.suffix == ".safetensors":
            # Load from safetensors
            state_dict = load_file(self.checkpoint_path)
            self.model.load_state_dict(state_dict)

            # Try to load metadata
            metadata_path = self.checkpoint_path.with_suffix(".json")
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self.class_names = metadata.get("class_names", self.CLASS_NAMES)
            else:
                self.class_names = self.CLASS_NAMES
        else:
            # Load from PyTorch checkpoint
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.class_names = checkpoint.get("class_names", self.CLASS_NAMES)

        self.model.to(self.device)
        self.model.eval()

        # Create transform for preprocessing
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)

        print(f"Loaded model from: {checkpoint_path}")
        print(f"Using device: {self.device}")

    def preprocess(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess an image."""
        # Load image (BGR)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale then back to 3-channel (match training)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Convert to tensor
        image = torch.from_numpy(image)

        # Apply timm normalization
        image = self.transform(image)

        return image

    @torch.no_grad()
    def predict(self, image_path: Path) -> dict:
        """
        Predict class probabilities for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with 'probabilities' and 'predicted_class'
        """
        # Preprocess
        image = self.preprocess(image_path)
        image = image.unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(image)

        # Get probabilities
        probs = F.softmax(logits, dim=1)[0]

        # Build result
        probabilities = {
            name: float(probs[i]) for i, name in enumerate(self.class_names)
        }
        predicted_idx = int(probs.argmax())
        predicted_class = self.class_names[predicted_idx]

        return {
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "predicted_idx": predicted_idx,
        }

    def predict_batch(self, paths: list[Path], output_file: Path = None) -> list[dict]:
        """
        Predict for multiple images and save results to file.

        Args:
            paths: List of image paths
            output_file: Path to save results (default: results.txt in first image's parent)

        Returns:
            List of prediction dicts
        """
        results = []

        for path in paths:
            try:
                pred = self.predict(path)
                results.append({
                    "path": path,
                    **pred
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({
                    "path": path,
                    "error": str(e)
                })

        # Determine output file
        if output_file is None and paths:
            output_file = paths[0].parent / "results.txt"

        # Write results to file
        if output_file:
            self._write_results(results, output_file)

        return results

    def _write_results(self, results: list[dict], output_file: Path):
        """Write results to a text file."""
        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Classification Results ({len(results)} images)\n")
            f.write("=" * 80 + "\n\n")

            for r in results:
                path = r["path"]

                if "error" in r:
                    f.write(f"File: {path.name}\n")
                    f.write(f"  ERROR: {r['error']}\n\n")
                    continue

                probs = r["probabilities"]
                predicted = r["predicted_class"]

                f.write(f"File: {path.name}\n")
                f.write(f"  Probabilities:\n")
                for name in self.CLASS_NAMES:
                    marker = " <--" if name == predicted else ""
                    f.write(f"    {name}: {probs[name]:.4f}{marker}\n")
                f.write(f"  Predicted: {predicted}\n")
                f.write("\n")

        print(f"\nResults saved to: {output_file}")


def find_images_in_folder(folder: Path) -> list[Path]:
    """Find all image files directly in a folder (non-recursive)."""
    images = set()
    for ext in ImageClassifier.IMAGE_EXTENSIONS:
        for p in folder.glob(f"*{ext}"):
            images.add(p)
        for p in folder.glob(f"*{ext.upper()}"):
            images.add(p)
    return sorted(images)


def find_folders_with_images(root: Path) -> list[Path]:
    """
    Recursively find all folders containing image files.

    Returns list of folder paths that contain at least one image.
    """
    folders_with_images = []
    root = Path(root)

    for folder in root.rglob("*"):
        if folder.is_dir():
            images = find_images_in_folder(folder)
            if images:
                folders_with_images.append(folder)

    return sorted(folders_with_images)


def find_images(path: Path) -> list[Path]:
    """Find all image files in a path (file or directory)."""
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() in ImageClassifier.IMAGE_EXTENSIONS:
            return [path]
        else:
            print(f"Warning: {path} is not a supported image format")
            return []

    if path.is_dir():
        return find_images_in_folder(path)

    print(f"Path not found: {path}")
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Classify images using trained MobileNetV4 model"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Single image file or folder containing images",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("v3/outputs/v1.pt"),
        help="Path to model checkpoint (default: v3/outputs/v1.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for results (default: results.txt in image directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find all folders with images and save results.txt in each",
    )
    args = parser.parse_args()

    # Initialize classifier
    classifier = ImageClassifier(args.checkpoint, device=args.device)

    # Handle recursive mode
    if args.recursive and args.path.is_dir():
        folders = find_folders_with_images(args.path)
        if not folders:
            print("No folders with images found.")
            sys.exit(1)

        print(f"Found {len(folders)} folder(s) with images\n")

        total_images = 0
        for folder in folders:
            images = find_images_in_folder(folder)
            if not images:
                continue

            print(f"\nProcessing: {folder} ({len(images)} images)")
            output_file = folder / "results.txt"
            results = classifier.predict_batch(images, output_file=output_file)
            total_images += len(images)

        print(f"\n{'='*50}")
        print(f"Total: {total_images} images in {len(folders)} folders")
        print("=" * 50)
        return

    # Single file or single folder mode
    images = find_images(args.path)
    if not images:
        print("No images found to process.")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process\n")

    # Run inference
    results = classifier.predict_batch(images, output_file=args.output)

    # Print summary to console
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    for r in results:
        if "error" in r:
            print(f"  {r['path'].name}: ERROR")
        else:
            print(f"  {r['path'].name}: {r['predicted_class']}")


if __name__ == "__main__":
    main()