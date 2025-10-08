# SAM-Med3D Feature Quality Evaluation

## Overview

This module provides tools to evaluate the quality of SAM-Med3D features for tumor classification before running the full TabPFN/LoCalPFN pipeline. It adds a simple classification head on top of the 3D encodings to directly assess feature discriminability.

## Motivation

When your Med3Pipe pipeline (SAM-Med3D → TabPFN/LoCalPFN) is not performing well, it's important to diagnose where the issue lies:

1. **Are the SAM-Med3D features good?** ← This tool answers this question
2. Is TabPFN/LoCalPFN the bottleneck?
3. Is the data quality or labeling the issue?

By training a simple classification head directly on the SAM-Med3D features, you can quickly determine if the features capture meaningful information for your task.

## What It Does

The tool:
1. Loads a pretrained SAM-Med3D model
2. Adds a lightweight classification head (Global Average Pooling + Dropout + Linear)
3. Trains the head (optionally fine-tuning the encoder)
4. Evaluates on validation set with metrics: Accuracy, AUC, Confusion Matrix

## Architecture

```
Input Volume (1, D, H, W)
        ↓
SAM-Med3D Image Encoder [frozen or trainable]
        ↓
Feature Map (C, d, h, w)  [e.g., C=768 for ViT-B]
        ↓
Global Average Pooling
        ↓
Flatten (C,)
        ↓
Dropout (0.3)
        ↓
Linear Classifier (C → 2)
        ↓
Logits (benign, malignant)
```

## Usage

### Option 1: Python Script

```bash
# Test with frozen encoder (recommended first)
python scripts/test_sam_features.py --dataset gist --epochs 10 --freeze

# Test with fine-tuning
python scripts/test_sam_features.py --dataset lipo --epochs 20 --no-freeze

# Custom configuration
python scripts/test_sam_features.py \
    --dataset gist \
    --dataset-root data/gist \
    --checkpoint path/to/sam_med3d.pth \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-3 \
    --freeze \
    --output-dir results/sam_eval
```

**Script Arguments:**
- `--dataset`: Dataset name (e.g., 'gist', 'lipo')
- `--dataset-root`: Path to dataset (default: `data/<dataset>`)
- `--checkpoint`: Path to SAM-Med3D pretrained weights
- `--freeze`: Freeze encoder, train only classification head (recommended first)
- `--no-freeze`: Fine-tune encoder end-to-end
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-3)
- `--output-dir`: Where to save results

### Option 2: Jupyter Notebook

Open and run `notebooks/Test-SAM-Features.ipynb` for an interactive walkthrough with visualizations.

### Option 3: Programmatic API

```python
from pathlib import Path
from med3pipe.data.prepare import prepare_for_sam3d, split_validation
from med3pipe.sam.core import load_labels_from_sheet
from med3pipe.training import run_classification_head_experiment

# Prepare dataset
sam3d_root = Path("SAM-Med3D-main/SAM-Med3D-main")
prepared, paths = prepare_for_sam3d(
    dataset_root=Path("data/gist"),
    sam3d_root=sam3d_root,
    category="gist",
    ct_name="ct_GIST",
)

# Split validation
split_validation(paths, split_ratio=0.8, seed=2025, copy=True)

# Load labels
df, lab_map = load_labels_from_sheet(
    sheet_csv=Path("data/gist/sheet.csv"),
    dataset_name="GIST",
    subject_col="Subject",
    label_col="Diagnosis_binary",
    case_suffix="_CT",
)

# Run experiment
results = run_classification_head_experiment(
    paths=paths,
    lab_map=lab_map,
    sam3d_root=sam3d_root,
    checkpoint=None,  # or Path to checkpoint
    freeze_encoder=True,  # Train only classification head
    num_epochs=10,
    batch_size=4,
    learning_rate=1e-3,
    output_dir=Path("results/classification_head/gist"),
)

# Check results
print(f"Best AUC: {results['best_auc']:.4f}")
print(f"Final Accuracy: {results['final_metrics'].accuracy:.4f}")
```

## Interpreting Results

### Good Features (AUC > 0.7 with frozen encoder)
✅ **Interpretation**: SAM-Med3D learned meaningful representations for your task.

**Next Steps**:
- Proceed with TabPFN/LoCalPFN pipeline
- The features are discriminative enough
- Consider fine-tuning for marginal improvements

### Moderate Features (0.6 < AUC < 0.7)
⚠️ **Interpretation**: Features have some signal but limited discriminability.

**Next Steps**:
- TabPFN/LoCalPFN might extract more from these features
- Try fine-tuning SAM-Med3D on your data
- Check if you're using the right pretrained checkpoint
- Verify data quality and preprocessing

### Poor Features (AUC < 0.6 even with fine-tuning)
❌ **Interpretation**: SAM-Med3D features are not suitable for this task.

**Next Steps**:
- Try different pretrained weights (if available)
- Consider alternative feature extraction strategies
- Check data quality:
  - Are the images corrupted?
  - Are the labels correct?
  - Is the segmentation mask quality good?
- Consider training from scratch or using a different architecture

## Output Files

When you run the experiment, the following files are saved to `output_dir`:

```
output_dir/
├── best_model.pt              # Best model checkpoint
├── history.npy                # Training history (loss, acc, auc per epoch)
├── predictions.npy            # Validation predictions
├── targets.npy                # Validation ground truth
├── probabilities.npy          # Validation probabilities
└── summary.txt                # Text summary of results
```

If using the notebook, you also get visualizations:
- `training_curves.png`: Loss and accuracy curves
- `confusion_matrix.png`: Confusion matrix heatmap
- `roc_curve.png`: ROC curve

## Technical Details

### Classification Head Architecture

```python
class TumorClassificationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=2, dropout=0.3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)
```

### Training Details

- **Optimizer**: AdamW with weight decay
- **Loss**: CrossEntropyLoss
- **Scheduler**: CosineAnnealingLR
- **Default LR**: 1e-3
- **Default Weight Decay**: 1e-4
- **Default Dropout**: 0.3

### Memory Considerations

- **Frozen encoder**: ~6-8 GB GPU memory (batch_size=4)
- **Fine-tuning**: ~12-16 GB GPU memory (batch_size=4)

If you run out of memory:
- Reduce `--batch-size` to 2 or 1
- Use smaller `--img-size` (e.g., 96 instead of 128)
- Use CPU with `--device cpu` (much slower)

## Comparison with Full Pipeline

| Aspect | Classification Head | Full Pipeline (TabPFN/LoCalPFN) |
|--------|---------------------|----------------------------------|
| **Purpose** | Feature quality check | Final classification |
| **Training** | Supervised (end-to-end) | In-context learning / local adaptation |
| **Speed** | Fast (~10 epochs) | Fast (no training needed) |
| **Memory** | High (full volumes) | Low (precomputed features) |
| **Use Case** | Diagnostic tool | Production model |

## Example Results

### GIST Dataset (Example)
```
Epoch [10/10] - Time: 45.23s
===========================================================
Train Loss: 0.3421 | Train Acc: 0.8621
Val Loss:   0.4102 | Val Acc:   0.8250
Val AUC:    0.8734
✓ Best model saved (AUC: 0.8734)

Final Validation Metrics
===========================================================
Accuracy: 0.8250
AUC:      0.8734

Confusion Matrix:
[[35  5]
 [ 2 18]]

Classification Report:
              precision    recall  f1-score   support

      Benign       0.95      0.88      0.91        40
   Malignant       0.78      0.90      0.84        20

    accuracy                           0.88        60
```

**Interpretation**: AUC of 0.87 with frozen encoder indicates excellent feature quality. Proceed with TabPFN/LoCalPFN.

## Troubleshooting

### Issue: Out of Memory

**Solution**:
```bash
python scripts/test_sam_features.py --dataset gist --batch-size 2 --img-size 96
```

### Issue: Training doesn't converge

**Possible causes**:
1. Learning rate too high → Try `--lr 1e-4`
2. Not enough data → Check dataset size
3. Poor data quality → Verify images and labels

### Issue: Low AUC even with fine-tuning

**Possible causes**:
1. Bad pretrained weights → Try different checkpoint
2. Data quality issues → Verify preprocessing
3. Label noise → Check annotation quality
4. Task mismatch → SAM-Med3D may not be suitable

## Advanced Options

### Custom Model Architecture

```python
from med3pipe.training import SAMWithClassificationHead, TumorClassificationHead

# Build SAM model
sam_model = build_sam3d_model(...)

# Custom classification head
class CustomHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# Attach and train
model = SAMWithClassificationHead(sam_model, num_classes=2)
model.classifier = CustomHead(in_channels=768)
```

### Multi-class Classification

For multi-class tumor classification:

```python
results = run_classification_head_experiment(
    paths=paths,
    lab_map=lab_map_multiclass,  # {case_id: class_idx} where class_idx in [0, N-1]
    num_classes=6,  # Adjust in the implementation
    # ... other args
)
```

Note: Current implementation is binary. For multi-class, modify `TumorClassificationHead` to accept `num_classes` parameter.

## Integration with Med3Pipe Pipeline

After evaluating features:

1. **If features are good** (AUC > 0.7):
   ```bash
   python -m med3pipe multi-tabpfn --config configs/datasets.yaml
   ```

2. **If features need improvement** (AUC < 0.7):
   - Fine-tune SAM-Med3D first
   - Use the best checkpoint for feature extraction
   - Then run TabPFN/LoCalPFN

3. **If features are poor** (AUC < 0.6):
   - Consider alternative approaches (see hieracascade)
   - Check data quality
   - Try different architectures

## References

- SAM-Med3D paper: [SAM-Med3D: Segment Anything in Medical 3D](https://arxiv.org/abs/2310.15161)
- TabPFN: [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)

## See Also

- `docs/MULTI_DATASET.md`: Running full Med3Pipe pipeline
- `docs/hieracascade/TWO_CASCADES_EXPLAINED.md`: Alternative deep learning approach
- `med3pipe/README.md`: Med3Pipe API reference
