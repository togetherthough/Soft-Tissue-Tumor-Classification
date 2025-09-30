# GIST Dataset Guide for GeoTopo-STS

Quick guide for using GeoTopo-STS with your GIST CT dataset.

---

## 📁 Your Dataset Structure

```
Med3Tab-PFN/data/gist/
├── gist-001_CT/
│   └── NIFTI/
│       ├── image.nii.gz
│       └── mask.nii.gz
├── gist-002_CT/
│   └── NIFTI/
│       ├── image.nii.gz
│       └── mask.nii.gz
├── gist-003_CT/
│   └── NIFTI/
│       ├── image.nii.gz
│       └── mask.nii.gz
└── ...
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Discover Your Cases

```python
from geotopo_sts.gist_data_loader import discover_gist_cases

cases = discover_gist_cases('../data/gist')
print(f"Found {len(cases)} GIST cases")
```

### Step 2: Preprocess All Cases

```bash
python -m geotopo_sts.gist_data_loader \
    --root ../data/gist \
    --output ./preprocessed_gist \
    --config config.yaml \
    --workers 4 \
    --create-splits
```

This will:
- ✅ Resample to 1.5mm isotropic
- ✅ Apply CT windowing (-150 to 350 HU)
- ✅ Extract tumor + 10mm rim
- ✅ Generate surface meshes
- ✅ Compute persistent homology
- ✅ Create train/val/test splits

**Time**: ~5-10 minutes per case (depends on tumor size)

### Step 3: Add Labels

Edit `preprocessed_gist/labels.txt`:

```
# Format: case_id label
gist-001_CT 0
gist-002_CT 1
gist-003_CT 0
...
```

Labels can be:
- Binary: 0=benign, 1=malignant
- Multi-class: 0=low-risk, 1=intermediate, 2=high-risk
- Subtype codes: Match your classification scheme

---

## 🎯 Training

```bash
python -m geotopo_sts.train \
    --config config.yaml \
    --data ./preprocessed_gist \
    --output ./outputs/gist_exp1 \
    --device cuda
```

**Expected training time**: 2-4 hours for ~100 cases (200 epochs)

Monitor in real-time:
- Check `outputs/gist_exp1/history.json`
- Best model saved automatically

---

## 📊 Evaluation

```bash
python -m geotopo_sts.eval \
    --config config.yaml \
    --checkpoint ./outputs/gist_exp1/best_model.pth \
    --data ./preprocessed_gist \
    --output ./eval_gist \
    --ablations \
    --robustness \
    --embeddings
```

Outputs:
- `eval_gist/metrics.json` - Overall performance
- `eval_gist/confusion_matrix.png` - Per-class results
- `eval_gist/ablation_results.json` - Component importance
- `eval_gist/embeddings.npz` - For visualization

---

## 🔬 Interactive Exploration

Use the Jupyter notebook:

```bash
jupyter notebook QuickStart_GIST.ipynb
```

This lets you:
1. Load and visualize one GIST case
2. See mesh extraction in action
3. View topology features
4. Test model inference
5. Extract embeddings

---

## ⚙️ Configuration for GIST

The `config.yaml` is already set up for GIST CT data:

```yaml
data:
  root_dir: '../data/gist'
  modality: 'ct'
  image_filename: 'image.nii.gz'
  mask_filename: 'mask.nii.gz'

preprocessing:
  intensity:
    ct:
      window: [-150, 350]  # Soft tissue window
      normalization: 'zscore'
```

**Adjust if needed**:
- `crop_size: [160, 160, 160]` → Larger if tumors are big
- `target_vertices: 8000` → Lower (5000) if out of memory
- `rim_radius_mm: 10.0` → Change peritumoral rim size

---

## 📈 Expected Performance

Baseline estimates for GIST classification:

| Configuration | Accuracy | Macro F1 |
|--------------|----------|----------|
| Voxel only   | ~0.70    | ~0.65    |
| + Rim        | ~0.74    | ~0.69    |
| + Topology   | ~0.77    | ~0.72    |
| + Mesh       | ~0.79    | ~0.75    |
| **Full**     | **~0.82**| **~0.78**|

*Actual performance depends on dataset size and task difficulty*

---

## 🐛 Troubleshooting

### Issue: "No cases found"
**Solution**: Check your path. Should be `../data/gist` from geotopo_sts folder.

```python
from pathlib import Path
print(Path('../data/gist').resolve())
print(list(Path('../data/gist').glob('gist-*_CT')))
```

### Issue: "Mask file missing"
**Solution**: Check if your masks are named differently:
- `segmentation.nii.gz` instead of `mask.nii.gz`?
- Update `config.yaml`:

```yaml
data:
  mask_filename: 'segmentation.nii.gz'
```

### Issue: Out of memory during preprocessing
**Solution**: Process fewer cases in parallel:

```bash
python -m geotopo_sts.gist_data_loader \
    --workers 1 \  # Reduce from 4 to 1
    --root ../data/gist \
    --output ./preprocessed_gist
```

### Issue: Mesh extraction fails
**Solution**: Tumors might be too small or mask is noisy. Check preprocessing output:

```python
# In notebook, after loading case
print(f"Tumor volume: {mask.sum() * np.prod(spacing) / 1000:.2f} cm³")
```

If volume < 1 cm³, mesh might fail. Consider:
- Reducing `target_vertices` to 3000
- Increasing `smooth_kernel` to [5, 5, 5]

---

## 📊 GIST-Specific Tips

### 1. Class Imbalance
If you have imbalanced classes (e.g., mostly low-risk):

```yaml
training:
  loss:
    class_balance: 'sqrt_inv'  # or 'inverse'
```

### 2. Small Dataset (<50 cases)
Use aggressive data augmentation:

```yaml
augmentation:
  rotation_range: 30  # Increase from 25
  flip_prob: 0.5
  intensity_scale: [0.8, 1.2]
```

### 3. Multi-Site Data
If GIST cases from different hospitals, add domain adaptation:
- Train with site labels
- Use adversarial training (advanced)

### 4. Risk Stratification
For low/intermediate/high risk:

```yaml
model:
  head:
    n_classes: 3
    type: 'euclidean'
```

Or use hierarchical:

```yaml
model:
  head:
    type: 'hyperbolic'
    hierarchy:
      enabled: true
```

Then define:
```python
# In training
ancestors = {
    0: [],      # low-risk (root)
    1: [0],     # intermediate (child of low)
    2: [1, 0]   # high (child of intermediate)
}
```

---

## 💡 Integration with Med3Tab Pipeline

You can combine GeoTopo-STS with your existing TabPFN pipeline:

```python
# Extract embeddings from GeoTopo-STS
from geotopo_sts.eval import Evaluator

evaluator = Evaluator(model, test_loader, config)
embeddings = evaluator.extract_embeddings()

# Use as features for TabPFN
X = embeddings['fused']  # (N, 256) features
y = embeddings['targets']

from tabpfn import TabPFNClassifier
clf = TabPFNClassifier()
clf.fit(X, y)
```

This gives you the best of both worlds:
- Deep geometric/topological features from GeoTopo-STS
- Fast, accurate classification from TabPFN

---

## 📧 Next Steps

1. **Test on one case**: Run `QuickStart_GIST.ipynb`
2. **Preprocess all**: `python -m geotopo_sts.gist_data_loader`
3. **Add labels**: Edit `preprocessed_gist/labels.txt`
4. **Train**: `python -m geotopo_sts.train`
5. **Analyze**: Check ablations and embeddings

---

**Your GIST dataset is ready for GeoTopo-STS!** 🚀

For questions or issues specific to GIST data, see the main [README.md](README.md) or open an issue.
