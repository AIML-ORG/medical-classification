"""
Fine-tune a BERT sequence classifier on pre-extracted text: Prescription / Report / Others.

Expects CSV columns: text, label (0–2). Install e.g.:
  pip install transformers torch pandas scikit-learn tqdm matplotlib seaborn
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

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


@dataclass(frozen=True)
class TrainConfig:
    train_csv: str = "V4.0_TRAIN_FINAL_CLEANED.csv"
    test_csv: str = "V4.0_TEST_FINAL_CLEANED.csv"
    model_name: str = "bert-mini"
    local_model_dir: str = "./bert-mini"
    best_model_dir: str = "best_model_3class"
    archive_zip_basename: str = "Model_7"
    epochs: int = 4
    batch_size: int = 16
    max_len_train: int = 128
    max_len_inference: int = 512
    learning_rate: float = 1e-5
    num_labels: int = 3
    val_confidence_threshold: float = 0.75
    val_margin_threshold: float = 0.3
    inference_confidence_threshold: float = 0.7
    inference_margin_threshold: float = 0.3
    val_fraction: float = 0.2
    random_state: int = 42


LABEL_TO_NAME = {0: "Prescription", 1: "Report", 2: "Others"}
NAME_TO_LABEL = {"Prescription": 0, "Report": 1, "Others": 2}


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
    ) -> None:
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


def load_train_val_split(config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_train = pd.read_csv(config.train_csv)
    full_train = full_train.dropna(subset=["text", "label"])
    train_frame, val_frame = train_test_split(
        full_train,
        test_size=config.val_fraction,
        stratify=full_train["label"],
        random_state=config.random_state,
    )
    print(f"Train size: {len(train_frame)} | Validation size: {len(val_frame)}")
    print(train_frame["label"].value_counts())
    return train_frame, val_frame


def build_tokenizer_and_model(config: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.local_model_dir,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model


def build_dataloaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    tokenizer,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        TextClassificationDataset(train_frame, tokenizer, config.max_len_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TextClassificationDataset(val_frame, tokenizer, config.max_len_train),
        batch_size=config.batch_size,
    )
    return train_loader, val_loader


def class_weights_tensor(train_labels: pd.Series, device: torch.device, num_labels: int) -> torch.Tensor:
    sorted_class_ids = np.sort(train_labels.unique())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=sorted_class_ids,
        y=train_labels,
    )
    print(f"COUNTS: {train_labels.value_counts().to_dict()}")
    print("WEIGHTS:")
    for i in range(num_labels):
        print(f"   - [{i}] {LABEL_TO_NAME[i]}: {weights[i]:.2f}")
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch_index: int,
) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch_index + 1} Training")
    for batch in progress:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    config: TrainConfig,
    epoch_index: int,
) -> tuple[float, float]:
    model.eval()
    val_loss_total = 0.0
    correct = 0
    total = 0
    progress = tqdm(val_loader, desc=f"Epoch {epoch_index + 1} Validation")
    with torch.no_grad():
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            top_two_prob, top_two_idx = torch.topk(probabilities, k=2, dim=1)
            top_confidence = top_two_prob[:, 0]
            margin = top_two_prob[:, 0] - top_two_prob[:, 1]
            final_pred = torch.where(
                (top_confidence < config.val_confidence_threshold)
                | (margin < config.val_margin_threshold),
                torch.tensor(2, device=device),
                top_two_idx[:, 0],
            )
            loss = loss_fn(logits, labels)
            val_loss_total += loss.item()
            correct += (final_pred == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=loss.item())
    avg_loss = val_loss_total / len(val_loader)
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def run_training(
    model: nn.Module,
    tokenizer,
    train_frame: pd.DataFrame,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> None:
    weights = class_weights_tensor(train_frame["label"], device, config.num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"Loss function initialized with class weights on {device}")

    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        train_one_epoch(model, train_loader, device, optimizer, loss_fn, epoch)
        avg_val_loss, _val_acc = validate_epoch(
            model, val_loader, device, loss_fn, config, epoch
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(config.best_model_dir)
            tokenizer.save_pretrained(config.best_model_dir)
            print("Best model saved")

    grad_mean = model.classifier.weight.grad
    if grad_mean is not None:
        print(grad_mean.abs().mean())


def load_trained_model(model_dir: str, device: torch.device):
    if not os.path.exists(model_dir):
        print(f"ERROR: Folder not found at {model_dir}")
        return None, None
    print(f"Folder found. Files inside: {os.listdir(model_dir)}")
    try:
        print(f"Loading model onto: {device}")
        tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        model.to(device)
        model.eval()
        print("BERT model and tokenizer ready for testing.")
        return model, tokenizer
    except Exception as exc:
        print(f"Load failed: {exc}")
        return None, None


def tokenize_batch_text(
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
) -> dict:
    return tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)


def predict_argmax_batch(
    model: nn.Module,
    tokenizer,
    texts: pd.Series,
    device: torch.device,
    max_length: int,
) -> tuple[list[str], list[float]]:
    predictions: list[str] = []
    confidences: list[float] = []
    model.eval()
    with torch.no_grad():
        for text_value in tqdm(texts):
            inputs = tokenize_batch_text(tokenizer, str(text_value), device, max_length)
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, top_index = torch.max(probabilities, dim=1)
            predictions.append(LABEL_TO_NAME[top_index.item()])
            confidences.append(confidence.item())
    return predictions, confidences


def predict_with_gate_batch(
    model: nn.Module,
    tokenizer,
    texts: pd.Series,
    device: torch.device,
    max_length: int,
    config: TrainConfig,
) -> list[str]:
    out: list[str] = []
    model.eval()
    with torch.no_grad():
        for text_value in tqdm(texts):
            inputs = tokenize_batch_text(tokenizer, str(text_value), device, max_length)
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            top_two_prob, top_two_idx = torch.topk(probabilities, k=2, dim=1)
            top_confidence = top_two_prob[0][0].item()
            second_prob = top_two_prob[0][1].item()
            margin = top_confidence - second_prob
            if top_confidence < config.inference_confidence_threshold or margin < config.inference_margin_threshold:
                predicted_index = 2
            else:
                predicted_index = top_two_idx[0][0].item()
            out.append(LABEL_TO_NAME[predicted_index])
    return out


def plot_confidence_by_file(test_frame: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    figure, (ax_rx, ax_rep, ax_oth) = plt.subplots(3, 1, figsize=(18, 18))

    def plot_category(dataframe: pd.DataFrame, label_name: str, color: str, axis) -> None:
        cat = dataframe[dataframe["predicted_label"] == label_name].sort_values(by="confidence")
        if not cat.empty:
            sns.lineplot(
                data=cat,
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

    plot_category(test_frame, "Prescription", "teal", ax_rx)
    plot_category(test_frame, "Report", "coral", ax_rep)
    plot_category(test_frame, "Others", "gray", ax_oth)
    plt.tight_layout()
    plt.show()


def plot_confusion_heatmap(test_frame: pd.DataFrame, overall_accuracy_percent: float) -> None:
    predicted_numeric = test_frame["predicted_label"].map(NAME_TO_LABEL)
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


def print_directory_size_megabytes(path: str) -> None:
    total_bytes = 0
    if not os.path.exists(path):
        print(f"Folder '{path}' not found.")
        return
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_bytes += os.path.getsize(filepath)
    print(f"Total disk size: {total_bytes / (1024 * 1024):.2f} MB")


def run_evaluation_and_plots(
    model: nn.Module,
    tokenizer,
    test_frame: pd.DataFrame,
    device: torch.device,
    config: TrainConfig,
) -> None:
    print(f"Running inference (argmax confidence) on {device}...")
    preds, confs = predict_argmax_batch(
        model, tokenizer, test_frame["text"], device, config.max_len_inference
    )
    test_frame = test_frame.copy()
    test_frame["predicted_label"] = preds
    test_frame["confidence"] = confs
    test_frame["true_label_name"] = test_frame["label"].map(LABEL_TO_NAME)
    plot_confidence_by_file(test_frame)

    print(f"Running inference with safety gate on {device}...")
    gated = predict_with_gate_batch(
        model, tokenizer, test_frame["text"], device, config.max_len_inference, config
    )
    test_frame["predicted_label"] = gated
    test_frame["true_label_name"] = test_frame["label"].map(LABEL_TO_NAME)

    print("\n--- SAMPLE PREDICTIONS ---")
    print(test_frame[["filename", "true_label_name", "predicted_label"]].head(400))

    results_csv = "MODEL_TEST_RESULTS.csv"
    test_frame.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")

    predicted_numeric = test_frame["predicted_label"].map(NAME_TO_LABEL)
    overall_accuracy_percent = accuracy_score(test_frame["label"], predicted_numeric) * 100
    plot_confusion_heatmap(test_frame, overall_accuracy_percent)

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

    print_directory_size_megabytes("best_model")
    parameter_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (parameter_bytes + buffer_bytes) / 1024**2
    print(f"Model memory size: {model_size_mb:.2f} MB")
    shutil.make_archive(config.archive_zip_basename, "zip", config.best_model_dir)


def main() -> None:
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_frame, val_frame = load_train_val_split(config)
    tokenizer, model = build_tokenizer_and_model(config)
    train_loader, val_loader = build_dataloaders(train_frame, val_frame, tokenizer, config)
    model.to(device)

    run_training(model, tokenizer, train_frame, train_loader, val_loader, device, config)

    model, tokenizer = load_trained_model(config.best_model_dir, device)
    if model is None or tokenizer is None:
        return

    if not os.path.exists(config.test_csv):
        print(f"Error: {config.test_csv} not found.")
        return

    test_frame = pd.read_csv(config.test_csv)
    print(f"Loaded {len(test_frame)} testing rows.")
    model.to(device)
    run_evaluation_and_plots(model, tokenizer, test_frame, device, config)


if __name__ == "__main__":
    main()
