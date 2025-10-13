# HieraCascade

**Hierarchical Cascaded Multi-Scale 3D Transformer for Soft Tissue Tumor Classification**

A mask-free two-stage deep learning architecture for multi-class classification of soft tissue tumors from CT/MRI scans.

## Architecture Overview

### Stage 1: Coarse Predictions + Saliency
- **Input**: Full body-part volume (192³ at 1.5mm isotropic spacing)
- **Backbone**: 3D Swin Transformer (Tiny/Small variant)
- **Outputs**:
  - Auxiliary classification logits for initial predictions
  - Dense saliency map for foreground detection
- **Purpose**: Generate crop proposals for the second stage

### Stage 2: Fine-Grained Hierarchical Classification
- **Input**: K=8 high-resolution crops (96³) selected from saliency peaks
- **Backbone**: 3D Swin Transformer (Base variant)
- **Outputs**:
  - Fine-grained class predictions (N classes)
  - Coarse family predictions (F families)
- **Loss**: Hierarchical consistency between fine and coarse predictions

## Key Features

# Mask-free: No segmentation masks required  
# Multi-modal: Handles both CT and MRI modalities  
# Hierarchical: Learns coarse→fine class relationships  
# Efficient: Cascaded architecture reduces computational cost  
# Robust: Site-balanced sampling handles domain shift  

## Installation

```bash
{{ ... }}
# Navigate to project directory and install dependencies
cd Med3Tab-PFN
pip install -r hieracascade/requirements.txt
```

### Requirements
- Python 3.8 or higher
- PyTorch 2.0+ with CUDA 11.8+
- nibabel, scipy, scikit-learn, matplotlib, pyyaml, tqdm

## Quick Start

### 1. Prepare Data

Expected directory structure:
```
data/
├── melanoma/
│   ├── Melanoma-001_CT/
│   │   └── 1/NIFTI/image.nii.gz
│   └── Melanoma-002_CT/
├── crlm/
├── gist/
├── lipo/
├── desmoid/
└── liver/
```

### 2. Select Label Column

The pipeline supports different label configurations from your `sheet.csv`:

- **`Diagnosis`** - Multi-class classification (melanoma, crlm, gist, lipo, desmoid, liver)
- **`Diagnosis_binary`** - Binary classification (malignant, benign)

Create labels CSV from sheet.csv:
```bash
python -m hieracascade.prepare_data \
    --data_root data \
    --output_csv labels.csv
```

**Or use quick start with label selection:**
```bash
# Multi-class (default)
python -m hieracascade.quick_start \
    --label_column Diagnosis \
    --data_root data \
    --sheet_csv data/sheet.csv

# Binary classification
python -m hieracascade.quick_start \
    --label_column Diagnosis_binary \
    --data_root data \
    --sheet_csv data/sheet.csv
```

See [LABEL_SELECTION_GUIDE.md](LABEL_SELECTION_GUIDE.md) for detailed instructions.

### 3. Train Stage 1

```bash
python -m hieracascade.train_stage1 \
    --config hieracascade/configs/stage1.yaml \
    --data_root data \
    --labels_csv labels.csv \
    --output_dir outputs/stage1/fold0 \
    --fold 0
```

### 3. Train Stage 2

```bash
python -m hieracascade.train_stage2 \
    --config hieracascade/configs/stage2.yaml \
    --data_root data \
    --labels_csv labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0 \
    --fold 0
```

### 4. Evaluate

```bash
python -m hieracascade.evaluate \
    --checkpoint outputs/stage2/fold0/checkpoint_best.pt \
    --stage stage2 \
    --data_root data \
    --labels_csv labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0/eval \
    --fold 0
```

## Configuration

### Stage 1 Config (`configs/stage1.yaml`)

Key parameters:
- `model.embed_dim`: Swin Transformer embedding dimension (96)
- `train.lam_sparsity`: Saliency sparsity weight (1e-3)
- `train.batch_size`: Batch size per GPU (2 for 192³)
- `geom.size`: Input volume dimensions [192, 192, 192]
- `geom.spacing`: Target isotropic spacing (1.5mm)

### Stage 2 Config (`configs/stage2.yaml`)

Key parameters:
- `model.pool`: Pooling strategy (set_transformer, attention_mil)
- `proposals.K`: Number of crops per study (8)
- `proposals.crop_size`: Crop dimensions (96)
- `train.alpha`: Coarse loss weight (0.3)
- `train.beta`: Consistency loss weight (0.1)

## Class Hierarchy

```
Fine Classes (N=6):
├── melanoma ──────→ malignant (coarse)
├── crlm ──────────→ malignant
├── gist ──────────→ malignant
├── lipo ──────────→ benign
├── desmoid ───────→ benign
└── liver ─────────→ other
```

The hierarchy can be modified in `hieracascade/dataio/utils.py`:
- `CLASS_HIERARCHY`: fine → coarse mapping
- `FINE_TO_IDX`: fine class names → indices
- `COARSE_TO_IDX`: coarse family names → indices

## Cross-Validation

Site-held-out K-fold cross validation (one site held out per fold):

```bash
# Fold 0
python -m hieracascade.train_stage1 --fold 0 ...
# Fold 1
python -m hieracascade.train_stage1 --fold 1 ...
```

## Outputs

### Training
- `checkpoint_best.pt`: Best model by validation F1 score
- `checkpoint_latest.pt`: Latest epoch checkpoint
- `config.yaml`: Training configuration used
- `plots/training_curves.png`: Loss and metrics over epochs
- `visualizations/`: Saliency overlays (Stage 1 only)
- `cache/`: Preprocessed volumes for faster loading

### Evaluation
- `predictions_fine.csv`: Per-study predictions with probabilities
- `confusion_matrix_fine.png`: Confusion matrix visualization
- `predictions_coarse.csv`: Family-level predictions
- Metrics: Macro-F1, Balanced Accuracy, AUC, per-class F1

## Advanced Usage

### Custom Class Weights

Enable in config for imbalanced datasets:
```yaml
train:
  use_class_weights: true
  focal_gamma: 2.0  # For Focal Loss
```

### Ablations

Modify configs for ablation studies:
- **Crop count**: `proposals.K` ∈ {4, 8, 12}
- **Crop size**: `proposals.crop_size` ∈ {80, 96, 112}
- **Pooling**: `model.pool` ∈ {set_transformer, attention_mil, gated_attention_mil}
- **Hierarchy**: Set `alpha=0, beta=0` to disable

### Joint Fine-Tuning

After Stage 2 converges, unfreeze Stage 1 last blocks for end-to-end training:
```python
# In train_stage2.py, after loading Stage 1:
for param in stage1_model.layers[-1].parameters():
    param.requires_grad = True
```

## Performance Tips

### GPU Memory
- **Stage 1** (192³): ~10-12GB per sample → batch_size=2 on 24GB GPU
- **Stage 2** (8×96³): ~8-10GB per study → batch_size=1-2

To reduce memory usage:
- Lower `geom.size` to [160, 160, 160]
- Reduce `proposals.K` to 4-6
- Use `torch.utils.checkpoint` for gradient checkpointing

### Speed
- Mixed precision training enabled by default via `autocast`
- Cache preprocessed volumes using `cache_dir` parameter
- Multi-GPU training possible with `torch.nn.DataParallel` or DDP

## Citation

If you use this implementation, please cite the original methods:

**Swin Transformer**:
```bibtex
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and others},
  booktitle={ICCV},
  year={2021}
}
```

**Set Transformer**:
```bibtex
@inproceedings{lee2019set,
  title={Set transformer: A framework for attention-based permutation-invariant neural networks},
  author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and others},
  booktitle={ICML},
  year={2019}
}
```

## Troubleshooting

**Out of memory**: Reduce batch size or volume size  
**Poor saliency**: Increase `lam_sparsity` or train longer  
**Low accuracy**: Check class balance, enable class weights  
**No crops found**: Increase `proposals.top_p` or decrease `nms_dist`

## License

MIT License - see project root for details.
