# MobileNetV4 3-class Fine-tuning Results

## Training Summary

**Model**: `mobilenetv4_conv_small.e3600_r256_in1k`
**Total Epochs**: 17 (early stopping triggered)
**Training Time**: ~6.9 hours
**Best Validation Loss**: 0.0089 (Epoch 14)

## Final Test Results

| Metric | Value |
|--------|-------|
| Test Loss | 0.0087 |
| Test Accuracy | 99.65% |

### Per-class Metrics (Test Set)

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| report | 0.992 | 0.994 | 0.993 |
| prescription | 0.997 | 0.996 | 0.996 |
| other | 1.000 | 1.000 | 1.000 |

## Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Checkpoint |
|-------|------------|-----------|----------|---------|------------|
| 1 | 0.4871 | 86.76% | 0.1027 | 96.49% | ✓ |
| 2 | 0.1645 | 93.96% | 0.1830 | 93.60% | - |
| 3 | 0.1565 | 94.90% | 0.0505 | 98.42% | ✓ |
| 4 | 0.0799 | 97.17% | 0.0439 | 98.77% | ✓ |
| 5 | 0.0755 | 97.22% | 0.0296 | 99.05% | ✓ |
| 6 | 0.0668 | 97.61% | 0.0312 | 98.97% | - |
| 7 | 0.0584 | 97.85% | 0.0332 | 98.62% | - |
| 8 | 0.0508 | 98.13% | 0.0216 | 99.27% | ✓ |
| 9 | 0.0382 | 98.58% | 0.0189 | 99.37% | ✓ |
| 10 | 0.0333 | 98.74% | 0.0172 | 99.35% | ✓ |
| 11 | 0.0319 | 98.76% | 0.0146 | 99.60% | ✓ |
| 12 | 0.0274 | 99.19% | 0.0313 | 99.05% | - |
| 13 | 0.0241 | 99.10% | 0.0114 | 99.62% | ✓ |
| 14 | 0.0227 | 99.14% | 0.0089 | 99.72% | ✓ |
| 15 | 0.0161 | 99.44% | 0.0094 | 99.70% | - |
| 16 | 0.0147 | 99.43% | 0.0097 | 99.72% | - |
| 17 | 0.0139 | 99.50% | 0.0097 | 99.72% | Early Stop |

## Dataset

- **Total images**: 26,557
- **Train**: 18,589 (70%)
- **Val**: 3,984 (15%)
- **Test**: 3,984 (15%)

### Class Distribution

| Class | Train | Val | Test |
|-------|-------|-----|------|
| report | 4,861 | 1,042 | 1,042 |
| prescription | 8,668 | 1,858 | 1,858 |
| other | 5,060 | 1,084 | 1,084 |

## Output Files

- `v3/outputs/best.pt` - Best model checkpoint
- `v3/outputs/training_history.json` - Training metrics per epoch
- `v3/outputs/test_results.json` - Final test results
- `v3/training_log.txt` - Full training log

## How to Use the Model

```python
import torch
import timm

# Load the checkpoint
checkpoint = torch.load("v3/outputs/best.pt", weights_only=False)

# Create model
model = timm.create_model("mobilenetv4_conv_small.e3600_r256_in1k", pretrained=False, num_classes=3)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Class mapping
class_names = checkpoint["class_names"]  # ["report", "prescription", "other"]
```