# Quick Feature Test - TL;DR

## Why?

Your Med3Pipe pipeline isn't working well. Before diving deep, check if SAM-Med3D features are any good.

## How? (30 seconds)

```bash
# Test GIST dataset with frozen encoder (10 epochs, ~10 min on GPU)
python scripts/test_sam_features.py --dataset gist --epochs 10 --freeze
```

## What do the results mean?

After training completes, check the **Validation AUC**:

### ✅ AUC > 0.7 (Good)
**Your SAM-Med3D features are fine!**
- The problem is NOT the features
- Continue with your TabPFN/LoCalPFN pipeline
- Maybe tune hyperparameters or check other parts

### ⚠️ AUC 0.6-0.7 (Moderate)
**Features have some signal but limited**
- Try fine-tuning SAM-Med3D: `--no-freeze --epochs 20`
- Check if you have the right checkpoint
- Verify preprocessing and data quality
- TabPFN/LoCalPFN might still help

### ❌ AUC < 0.6 (Poor)
**Features are not discriminative for your task**
- SAM-Med3D features won't work well
- Check data quality:
  - Are images preprocessed correctly?
  - Are labels correct?
  - Are masks good quality?
- Consider:
  - Different pretrained checkpoint
  - Alternative architecture (see hieracascade)
  - Training from scratch

## Alternative: Use Notebook

Open `notebooks/Test-SAM-Features.ipynb` for interactive walkthrough with plots.

## Output Location

Results saved to: `results/classification_head/<dataset>/`

Key files:
- `summary.txt` - Quick metrics summary
- `best_model.pt` - Trained model checkpoint
- Notebook also generates plots (ROC curve, confusion matrix, training curves)

## Example Output

```
===========================================================
Training Complete!
===========================================================
Best Epoch: 8 | Best Val AUC: 0.8234

===========================================================
Final Validation Metrics
===========================================================
Accuracy: 0.8000
AUC:      0.8234

Confusion Matrix:
[[32  8]
 [ 4 16]]
```

**Interpretation**: AUC of 0.82 is good! SAM-Med3D features work for this task.

## Common Issues

**Out of memory?**
```bash
python scripts/test_sam_features.py --dataset gist --batch-size 2
```

**Takes too long?**
```bash
python scripts/test_sam_features.py --dataset gist --epochs 5
```

**CPU only?**
```bash
python scripts/test_sam_features.py --dataset gist --device cpu --epochs 5
```

## Full Documentation

See `docs/SAM_FEATURE_EVALUATION.md` for detailed guide.
