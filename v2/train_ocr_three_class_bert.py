"""
Fine-tune a BERT sequence classifier for OCR text: Prescription / Report / Others.

Install dependencies separately, e.g.:
  pip install transformers==4.37.0 torch torchvision easyocr pillow pandas scikit-learn tqdm matplotlib seaborn
"""

from __future__ import annotations

import os
import shutil
import site
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)


def _ensure_user_site_on_path() -> None:
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.insert(0, user_site)


_ensure_user_site_on_path()

TRAIN_CSV = "V4.0_TRAIN_FINAL_CLEANED.csv"
TEST_CSV = "V4.0_TEST_FINAL_CLEANED.csv"
MODEL_NAME = "bert-mini"
LOCAL_MODEL_DIR = "./bert-mini"
BEST_MODEL_DIR = "best_model_3class"
ARCHIVE_ZIP_BASENAME = "Model_7"
EPOCHS = 4
BATCH_SIZE = 16
MAX_LEN_TRAIN = 128
MAX_LEN_INFERENCE = 512
LEARNING_RATE = 1e-5
NUM_LABELS = 3

VAL_CONFIDENCE_THRESHOLD = 0.75
VAL_MARGIN_THRESHOLD = 0.3

INFERENCE_CONFIDENCE_THRESHOLD = 0.7
INFERENCE_MARGIN_THRESHOLD = 0.3

LABEL_TO_NAME = {0: "Prescription", 1: "Report", 2: "Others"}
NAME_TO_LABEL = {"Prescription": 0, "Report": 1, "Others": 2}


class OcrTextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = MAX_LEN_TRAIN):
        self._frame = dataframe
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int):
        text = str(self._frame.iloc[index]["text"])
        label = int(self._frame.iloc[index]["label"])

        encoded = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if "token_type_ids" in item:
            del item["token_type_ids"]

        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def main() -> None:
    full_train_frame = pd.read_csv(TRAIN_CSV)
    full_train_frame = full_train_frame.dropna(subset=["text", "label"])

    train_frame, validation_frame = train_test_split(
        full_train_frame,
        test_size=0.2,
        stratify=full_train_frame["label"],
        random_state=42,
    )

    print(f"Train size: {len(train_frame)} | Validation size: {len(validation_frame)}")
    print(train_frame["label"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_DIR,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    train_loader = DataLoader(
        OcrTextDataset(train_frame, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        OcrTextDataset(validation_frame, tokenizer),
        batch_size=BATCH_SIZE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_frame = pd.read_csv(TRAIN_CSV)
    print(train_frame["label"].unique())
    print(train_frame["label"].min(), train_frame["label"].max())

    print("Unique labels in training data:", train_frame["label"].unique())
    print("Count of each label:\n", train_frame["label"].value_counts())

    sorted_class_ids = np.sort(train_frame["label"].unique())
    sklearn_class_weights = compute_class_weight(
        class_weight="balanced",
        classes=sorted_class_ids,
        y=train_frame["label"],
    )

    print(f"COUNTS: {train_frame['label'].value_counts().to_dict()}")
    print("WEIGHTS:")
    print(f"   - [0] Prescription  : {sklearn_class_weights[0]:.2f}")
    print(f"   - [1] Report        : {sklearn_class_weights[1]:.2f} ")
    print(f"   - [2] Others       : {sklearn_class_weights[2]:.2f} ")

    class_weights_tensor = torch.tensor(sklearn_class_weights, dtype=torch.float).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print(f"Loss function initialized with class weights on {device}")

    best_validation_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")

        for batch in train_progress:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_progress.set_postfix(loss=loss.item())

        average_train_loss = train_loss_total / len(train_loader)

        model.eval()
        validation_loss_total = 0.0
        correct_predictions = 0
        total_labels = 0

        validation_progress = tqdm(validation_loader, desc=f"Epoch {epoch + 1} Validation")

        with torch.no_grad():
            for batch in validation_progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                probabilities = torch.softmax(logits, dim=1)
                top_two_probabilities, top_two_indices = torch.topk(probabilities, k=2, dim=1)

                top_confidence = top_two_probabilities[:, 0]
                margin = top_two_probabilities[:, 0] - top_two_probabilities[:, 1]

                final_predictions = torch.where(
                    (top_confidence < VAL_CONFIDENCE_THRESHOLD) | (margin < VAL_MARGIN_THRESHOLD),
                    torch.tensor(2).to(device),
                    top_two_indices[:, 0],
                )

                loss = loss_fn(logits, labels)
                validation_loss_total += loss.item()

                correct_predictions += (final_predictions == labels).sum().item()
                total_labels += labels.size(0)

                validation_progress.set_postfix(loss=loss.item())
                average_validation_loss = validation_loss_total / len(validation_loader)
                validation_accuracy = correct_predictions / total_labels

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            print("Best model saved")

    print(model.classifier.weight.grad.abs().mean())

    model_path = BEST_MODEL_DIR
    if not os.path.exists(model_path):
        print(f"ERROR: Folder not found at {model_path}")
    else:
        print(f"Folder found. Files inside: {os.listdir(model_path)}")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading model onto: {device}")

            tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
            model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)

            model.to(device)
            model.eval()
            print("BERT model and tokenizer ready for testing.")
        except Exception as exc:
            print(f"Load failed: {exc}")

    if not os.path.exists(TEST_CSV):
        print(f"Error: {TEST_CSV} not found.")
        return

    test_frame = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_frame)} testing rows.")

    model.to(device)

    predictions_for_plot: list[str] = []
    confidence_scores: list[float] = []

    print(f"Running inference (argmax confidence) on {device}...")

    model.eval()
    with torch.no_grad():
        for text_value in tqdm(test_frame["text"]):
            inputs = tokenizer(
                str(text_value),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN_INFERENCE,
            ).to(device)

            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, top_index = torch.max(probabilities, dim=1)

            predictions_for_plot.append(LABEL_TO_NAME[top_index.item()])
            confidence_scores.append(confidence.item())

    test_frame["predicted_label"] = predictions_for_plot
    test_frame["confidence"] = confidence_scores
    test_frame["true_label_name"] = test_frame["label"].map(LABEL_TO_NAME)

    sns.set_style("whitegrid")
    figure, (axis_prescription, axis_report, axis_others) = plt.subplots(3, 1, figsize=(18, 18))

    def plot_category(dataframe: pd.DataFrame, label_name: str, color: str, axis) -> None:
        category_frame = dataframe[dataframe["predicted_label"] == label_name].sort_values(by="confidence")

        if not category_frame.empty:
            sns.lineplot(
                data=category_frame,
                x="filename",
                y="confidence",
                marker="o",
                color=color,
                ax=axis,
            )
            axis.set_title(f"Confidence per File: {label_name.upper()}", fontsize=14, fontweight="bold")
            axis.set_ylim(0, 1.05)
            axis.set_ylabel("Confidence Score")
            axis.set_xlabel("Filenames (Sorted by Confidence)")
            axis.tick_params(axis="x", rotation=90, labelsize=7)
        else:
            axis.set_title(f"No samples found for: {label_name.upper()}")

    plot_category(test_frame, "Prescription", "teal", axis_prescription)
    plot_category(test_frame, "Report", "coral", axis_report)
    plot_category(test_frame, "Others", "gray", axis_others)

    plt.tight_layout()
    plt.show()

    predictions_with_gate: list[str] = []

    print(f"Running inference with safety gate on {device}...")

    model.eval()
    with torch.no_grad():
        for text_value in tqdm(test_frame["text"]):
            inputs = tokenizer(
                str(text_value),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN_INFERENCE,
            ).to(device)

            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            top_two_probabilities, top_two_indices = torch.topk(probabilities, k=2, dim=1)

            top_confidence = top_two_probabilities[0][0].item()
            second_probability = top_two_probabilities[0][1].item()
            margin = top_confidence - second_probability

            if top_confidence < INFERENCE_CONFIDENCE_THRESHOLD or margin < INFERENCE_MARGIN_THRESHOLD:
                predicted_index = 2
            else:
                predicted_index = top_two_indices[0][0].item()

            predictions_with_gate.append(LABEL_TO_NAME[predicted_index])

    test_frame["predicted_label"] = predictions_with_gate
    test_frame["true_label_name"] = test_frame["label"].map(LABEL_TO_NAME)

    print("\n--- SAMPLE PREDICTIONS ---")
    print(test_frame[["filename", "true_label_name", "predicted_label"]].head(400))

    results_csv = "MODEL_TEST_RESULTS.csv"
    test_frame.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")

    predicted_numeric = test_frame["predicted_label"].map(NAME_TO_LABEL)
    overall_accuracy_percent = accuracy_score(test_frame["label"], predicted_numeric) * 100

    confusion = confusion_matrix(test_frame["label"], predicted_numeric)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Prescription", "Report", "Others"],
        yticklabels=["Prescription", "Report", "Others"],
    )
    plt.title(
        f"HQ Performance: 3-Class Document Classification\nOverall Accuracy: {overall_accuracy_percent:.2f}%",
        fontsize=16,
    )
    plt.ylabel("Actual Category (Ground Truth)", fontsize=12)
    plt.xlabel("AI Predicted Category", fontsize=12)
    plt.show()

    print(f"SUMMARY: The model achieved {overall_accuracy_percent:.2f}% accuracy across all document types.")
    print("\n--- Detailed Classification Report ---")
    print(
        classification_report(
            test_frame["label"],
            predicted_numeric,
            target_names=["Prescription", "Report", "Others"],
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    def print_directory_size_megabytes(path: str = "best_model") -> None:
        total_bytes = 0
        if not os.path.exists(path):
            print(f"Folder '{path}' not found.")
            return
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_bytes += os.path.getsize(filepath)
        print(f"Total disk size: {total_bytes / (1024 * 1024):.2f} MB")

    print_directory_size_megabytes("best_model")

    parameter_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (parameter_bytes + buffer_bytes) / 1024**2
    print(f"Model memory size: {model_size_mb:.2f} MB")

    shutil.make_archive(ARCHIVE_ZIP_BASENAME, "zip", BEST_MODEL_DIR)


if __name__ == "__main__":
    main()
