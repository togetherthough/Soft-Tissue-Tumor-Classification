# SAM-Med3D Feature Test - Cheat Sheet

## 🎯 Quick Commands

```bash
# Basic test (recommended first)
python scripts/test_sam_features.py --dataset gist --epochs 10 --freeze

# With fine-tuning
python scripts/test_sam_features.py --dataset gist --epochs 20 --no-freeze

# Low memory
python scripts/test_sam_features.py --dataset gist --batch-size 2 --img-size 96

# Fast test
python scripts/test_sam_features.py --dataset gist --epochs 5

# Minimal example
python examples/test_sam_features_minimal.py
```

## 📊 Result Interpretation

| AUC Score | Meaning | Action |
|-----------|---------|--------|
| **> 0.7** | ✅ Good features | Continue with TabPFN/LoCalPFN |
| **0.6-0.7** | ⚠️ Moderate | Try fine-tuning or check data |
| **< 0.6** | ❌ Poor | Investigate alternatives |

## 📁 Output Files

```
results/classification_head/<dataset>/
├── best_model.pt           # Best checkpoint
├── summary.txt             # Quick metrics
├── history.npy             # Training history
├── predictions.npy         # Val predictions
├── targets.npy             # Val targets
└── probabilities.npy       # Val probabilities
```

## 🔧 Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | Dataset name (e.g., 'gist') |
| `--epochs` | 10 | Number of training epochs |
| `--freeze` | True | Freeze encoder (train head only) |
| `--batch-size` | 4 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--checkpoint` | None | SAM-Med3D checkpoint path |
| `--device` | auto | 'cuda' or 'cpu' |

## 📚 Documentation

- **Quick Start**: `docs/QUICK_FEATURE_TEST.md`
- **Full Guide**: `docs/SAM_FEATURE_EVALUATION.md`
- **Main README**: `README.md` (updated)
- **Implementation**: `IMPLEMENTATION_COMPLETE.md`

## 🐍 Python API

```python
from med3pipe.training import run_classification_head_experiment

results = run_classification_head_experiment(
    paths=paths,
    lab_map=lab_map,
    freeze_encoder=True,
    num_epochs=10,
)
print(f"AUC: {results['best_auc']:.4f}")
```

## 📓 Jupyter Notebook

```bash
jupyter notebook notebooks/Test-SAM-Features.ipynb
```

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `--batch-size 2` or `--img-size 96` |
| Too slow | `--epochs 5` or use GPU |
| CPU only | `--device cpu --epochs 5` |
| Import error | Ensure `med3pipe` is in PYTHONPATH |

## 🔄 Workflow

```
1. Test features → python scripts/test_sam_features.py --dataset gist --freeze
   ↓
2. Check AUC
   ↓
   ├─ Good (>0.7)  → Continue with Med3Pipe pipeline
   ├─ Moderate     → Try fine-tuning (--no-freeze)
   └─ Poor (<0.6)  → Check data or try alternatives
```

## 🎓 Example Session

```bash
$ python scripts/test_sam_features.py --dataset gist --epochs 10 --freeze

[1/4] Preparing dataset...
✓ Prepared 120 cases

[2/4] Loading labels...
✓ Loaded 120 labels
  Benign: 75, Malignant: 45

[3/4] Building SAM-Med3D model...
✓ Model loaded

[4/4] Training...
Epoch [10/10] - Time: 45s
Train Loss: 0.34 | Train Acc: 0.86
Val Loss: 0.41 | Val Acc: 0.83
Val AUC: 0.87
✓ Best model saved

===========================================================
RESULTS
===========================================================
Best AUC: 0.8734
✅ GOOD: Features are discriminative!
```

## 💡 Pro Tips

1. **Always test frozen first** - If pretrained features work, no need to fine-tune
2. **Check AUC, not accuracy** - More robust metric for imbalanced data
3. **Use notebook for exploration** - Includes visualizations
4. **Save outputs** - Can analyze later without re-running
5. **Compare datasets** - Test on multiple datasets to identify patterns
