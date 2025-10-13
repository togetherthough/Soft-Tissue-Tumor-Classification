# HieraCascade Tutorial

Complete step-by-step guide to training and evaluating HieraCascade on your soft tissue tumor dataset.

## Two ways to Run

###  **Option 1: Terminal** (Recommended for Training)
Best for long training runs (hours). More stable, better GPU utilization.
```bash
python -m hieracascade.quick_start --label_column diagnosis --fold 0
```
**Follow this tutorial below** ↓

###  **Option 2: Jupyter Notebook** (Best for Exploration)
Best for interactive analysis, visualization, and understanding the pipeline.
```bash
# See: notebooks/hieracascade_full_pipeline.py
# Open in Jupyter or VS Code
```
**See:** `notebooks/README.md` for notebook instructions.

---

## Prerequisites

```bash
# Install dependencies
pip install -r hieracascade/requirements.txt

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Requirements
- Python 3.8 or higher
- PyTorch 2.0+ with CUDA 11.8+
- nibabel, scipy, scikit-learn, matplotlib, pyyaml, tqdm

## Data Preparation

### Expected Data Structure

```
data/
├── melanoma/
│   ├── Melanoma-001_CT/
│   │   └── 1/NIFTI/image.nii.gz
│   └── Melanoma-002_MR/
├── crlm/
├── gist/
├── lipo/
├── desmoid/
├── liver/
└── sheet.csv  # Labels file
```

### sheet.csv Format

Required columns:
- `case_id`: Study identifier
- `Diagnosis` or `Diagnosis_binary`: Class labels

Optional columns:
- `modality`: CT or MRI (inferred from case_id if missing)
- `site`: Hospital/site identifier (uses class if missing)

Example:
```csv
case_id,Diagnosis,Diagnosis_binary,modality,site
Melanoma-001,melanoma,malignant,CT,hospital_a
CRLM-045,crlm,malignant,CT,hospital_b
Desmoid-023,desmoid,benign,MRI,hospital_a
Lipo-078,lipo,benign,CT,hospital_c
```

## Quick Start (Recommended)

Run the complete pipeline with a single command:

```bash
python -m hieracascade.quick_start \
    --data_root data \
    --sheet_csv data/sheet.csv \
    --output_dir outputs/hieracascade \
    --fold 0 \
    --device cuda
```

This will:
1. ✅ Load labels from sheet.csv
2. ✅ Train Stage-1 (coarse + saliency)
3. ✅ Train Stage-2 (hierarchical classification)
4. ✅ Save all checkpoints and visualizations

## Step-by-Step Training

### Step 1: Prepare Labels

```bash
# Option A: From sheet.csv (recommended)
python -c "
from hieracascade.dataio import create_index_from_sheet
create_index_from_sheet(
    data_root='data',
    sheet_path='data/sheet.csv',
    output_csv='data/labels.csv'
)
"

# Option B: Scan directory structure
python -m hieracascade.prepare_data \
    --data_root data \
    --output_csv data/labels.csv
```

Output: `data/labels.csv` with processed labels

### Step 2: Train Stage-1

Stage-1 learns coarse predictions and generates saliency maps for crop proposals.

```bash
python -m hieracascade.train_stage1 \
    --config hieracascade/configs/stage1.yaml \
    --data_root data \
    --labels_csv data/labels.csv \
    --output_dir outputs/stage1/fold0 \
    --fold 0 \
    --device cuda
```

**Key Parameters** (edit `configs/stage1.yaml`):
- `train.epochs`: 20 (default)
- `train.batch_size`: 2 (for 192³ volumes on 24GB GPU)
- `train.lam_sparsity`: 1e-3 (saliency sparsity weight)
- `geom.size`: [192, 192, 192] (input dimensions)

**Outputs**:
- `checkpoint_best.pt`: Best model by validation F1
- `visualizations/`: Saliency overlay images
- `plots/training_curves.png`: Training progress

**Expected Training Time**: ~3-5 hours on RTX 3090/A100 for 50-100 studies

### Step 3: Train Stage-2

Stage-2 uses Stage-1 saliency to extract crops and performs hierarchical classification.

```bash
python -m hieracascade.train_stage2 \
    --config hieracascade/configs/stage2.yaml \
    --data_root data \
    --labels_csv data/labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0 \
    --fold 0 \
    --device cuda
```

**Key Parameters** (edit `configs/stage2.yaml`):
- `train.epochs`: 40 (default)
- `proposals.K`: 8 (crops per study)
- `proposals.crop_size`: 96 (crop dimensions)
- `train.alpha`: 0.3 (coarse loss weight)
- `train.beta`: 0.1 (consistency loss weight)

**Outputs**:
- `checkpoint_best.pt`: Best model
- `plots/training_curves.png`: Training curves

**Expected Training Time**: ~5-8 hours on RTX 3090/A100

## Evaluation

### Evaluate Stage-2

```bash
python -m hieracascade.evaluate \
    --checkpoint outputs/stage2/fold0/checkpoint_best.pt \
    --stage stage2 \
    --data_root data \
    --labels_csv data/labels.csv \
    --stage1_ckpt outputs/stage1/fold0/checkpoint_best.pt \
    --output_dir outputs/stage2/fold0/eval \
    --fold 0 \
    --device cuda
```

**Outputs**:
- `confusion_matrix_fine.png`: Fine-grained confusion matrix
- `confusion_matrix_coarse.png`: Coarse family confusion matrix
- `predictions_fine.csv`: Per-study predictions with probabilities
- Console: Macro-F1, Balanced Accuracy, AUC, per-class metrics

### Evaluate Stage-1 (Optional)

```bash
python -m hieracascade.evaluate \
    --checkpoint outputs/stage1/fold0/checkpoint_best.pt \
    --stage stage1 \
    --data_root data \
    --labels_csv data/labels.csv \
    --output_dir outputs/stage1/fold0/eval \
    --fold 0
```

## Cross-Validation

Run multiple folds for robust evaluation:

```bash
# Train all folds
for fold in 0 1 2 3; do
    # Stage-1
    python -m hieracascade.train_stage1 \
        --config hieracascade/configs/stage1.yaml \
        --data_root data \
        --labels_csv data/labels.csv \
        --output_dir outputs/stage1/fold$fold \
        --fold $fold
    
    # Stage-2
    python -m hieracascade.train_stage2 \
        --config hieracascade/configs/stage2.yaml \
        --data_root data \
        --labels_csv data/labels.csv \
        --stage1_ckpt outputs/stage1/fold$fold/checkpoint_best.pt \
        --output_dir outputs/stage2/fold$fold \
        --fold $fold
done

# Aggregate results
python -m hieracascade.aggregate_cv_results \
    --results_dir outputs/stage2 \
    --n_folds 4
```

## Configuration Tuning

### For Class Imbalance

Enable class weighting and focal loss:

```yaml
# In configs/stage2.yaml
train:
  use_class_weights: true
  focal_gamma: 2.0  # Focal loss
```

### For Limited GPU Memory

Reduce memory footprint:

```yaml
# In configs/stage1.yaml
geom:
  size: [160, 160, 160]  # Smaller volumes

train:
  batch_size: 1

# In configs/stage2.yaml
proposals:
  K: 4  # Fewer crops
  crop_size: 80  # Smaller crops
```

### For Better Saliency

If saliency maps are too sparse or too dense:

```yaml
# In configs/stage1.yaml
train:
  lam_sparsity: 5.0e-3  # Higher for more sparsity
  # or
  lam_sparsity: 1.0e-4  # Lower for less sparsity
```

## Ablation Studies

Test different configurations:

### Crop Count
```bash
for K in 4 8 12; do
    # Update proposals.K in config and retrain Stage-2
done
```

### Pooling Strategy
```yaml
model:
  pool: set_transformer  # or attention_mil, gated_attention_mil
```

### Hierarchy Weight
```yaml
train:
  alpha: 0.0  # Disable coarse loss
  beta: 0.0   # Disable consistency
```

## Inference on New Data

```python
import torch
from hieracascade import build_stage1_model, build_stage2_model
from hieracascade.dataio import preprocess_volume, get_crop_centers_from_saliency, extract_crop
import numpy as np

# Load models
device = 'cuda'
ckpt1 = torch.load('outputs/stage1/fold0/checkpoint_best.pt')
ckpt2 = torch.load('outputs/stage2/fold0/checkpoint_best.pt')

model1 = build_stage1_model(n_classes=6, backbone='swin3d_t', embed_dim=96).to(device)
model1.load_state_dict(ckpt1['model_state_dict'])
model1.eval()

model2 = build_stage2_model(n_fine=6, n_coarse=3, backbone='swin3d_b', embed_dim=768, pooling='set_transformer').to(device)
model2.load_state_dict(ckpt2['model_state_dict'])
model2.eval()

# Preprocess new volume
volume = preprocess_volume('path/to/new_scan.nii.gz', 'CT', target_spacing=1.5, target_size=(192,192,192))
volume_tensor = torch.from_numpy(volume[None, None]).float().to(device)
modality_id = torch.tensor([0]).to(device)  # 0=CT, 1=MRI

# Stage-1: Get saliency
with torch.no_grad():
    logits1, saliency = model1(volume_tensor, modality_id)
    
    # Extract crops
    centers = get_crop_centers_from_saliency(saliency[0,0], K=8)
    crops = [extract_crop(volume, c, 96) for c in centers]
    crops_tensor = torch.from_numpy(np.stack(crops)[:,None]).float().to(device).unsqueeze(0)
    
    # Stage-2: Predictions
    logits_fine, logits_coarse, _ = model2(crops_tensor, modality_id)
    
    # Get class
    pred = torch.argmax(logits_fine, dim=1).item()
    prob = torch.softmax(logits_fine, dim=1)[0, pred].item()
    
    classes = ['melanoma', 'crlm', 'gist', 'lipo', 'desmoid', 'liver']
    print(f"Prediction: {classes[pred]} (confidence: {prob:.3f})")
```

## Troubleshooting

### Out of Memory
- Reduce `train.batch_size` to 1
- Reduce `geom.size` to [160,160,160]
- Reduce `proposals.K` to 4-6

### Poor Saliency Quality
- Increase training epochs for Stage-1
- Adjust `train.lam_sparsity`
- Check data preprocessing (HU windows for CT)

### Low Accuracy
- Enable class weights: `use_class_weights: true`
- Use focal loss: `focal_gamma: 2.0`
- Increase training epochs
- Check data quality and labels

### No Crops Found
- Increase `proposals.top_p` (e.g., 0.01)
- Decrease `proposals.nms_dist` (e.g., 8)
- Check Stage-1 saliency quality

## Next Steps

1. **Experiment with ablations** (crop count, pooling, hierarchy)
2. **Analyze per-site performance** for domain generalization
3. **Export to ONNX** for deployment
4. **Ensemble multiple folds** for production

## Support

For issues or questions:
- Check `hieracascade/README.md` for architecture details
- Review notebooks: `notebooks/HieraCascade-Demo.ipynb`
- See example outputs in `outputs/` after training
