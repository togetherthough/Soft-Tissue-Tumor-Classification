# Full Pipeline Test Guide

This guide helps you run a complete end-to-end test of the SAM-Med3D pipeline with the new **Average Pooling** logic (replacing ROI pooling).

## Quick Test (10 cases, ~5-10 minutes)

Test with a small subset to verify everything works:

```bash
python test_full_pipeline.py --dataset-root gist --max-cases 10
```

## Full Test (All cases)

Run on the complete dataset:

```bash
python test_full_pipeline.py --dataset-root gist --max-cases 999
```

## Custom Options

```bash
python test_full_pipeline.py \
  --dataset-root gist \
  --category gist \
  --ct-name ct_GIST \
  --max-cases 20 \
  --split-ratio 0.8 \
  --seed 2025 \
  --device cuda  # or 'cpu' for CPU-only
```

## What the Test Does

The script runs through all 8 steps of the pipeline:

1. **Prepare Data**: Discovers and prepares cases into SAM-Med3D format
2. **Split Validation**: Creates train/val split
3. **Build Model**: Loads SAM-Med3D model (with checkpoint if available)
4. **Extract Embeddings**: Runs image encoder on all volumes
5. **Pool Features**: Applies **Global Average Pooling** (NEW - not ROI pooling)
6. **Load Labels**: Reads labels from sheet.csv
7. **Preprocess**: StandardScaler + PCA
8. **Train TabPFN**: Trains and evaluates TabPFN classifier

## Expected Output

If successful, you'll see:

```
============================================================
✅ PIPELINE TEST COMPLETED SUCCESSFULLY!
============================================================

Results saved to: test_pipeline_results/gist_ct_GIST_test

Metrics:
  Accuracy: 0.XXXX
  AUC:      0.XXXX

Confusion Matrix:
[[TN FP]
 [FN TP]]
```

## Verification Points

The test specifically verifies:

1. ✅ **Average pooling is used** (not ROI pooling with masks)
2. ✅ Embeddings are extracted successfully
3. ✅ Feature shapes are consistent
4. ✅ Labels align correctly
5. ✅ TabPFN training completes
6. ✅ Predictions are generated

## Troubleshooting

### Issue: "Dataset root not found"
**Solution**: Make sure you have the `gist` folder in your project directory, or specify the correct path with `--dataset-root`

### Issue: "No checkpoints found"
**Solution**: The script will work with random weights for testing. For better results, ensure `sam-med3d/ckpt/sam_med3d_turbo.pth` exists.

### Issue: "sheet.csv not found"
**Solution**: Ensure `sheet.csv` is in the dataset root or project root directory.

### Issue: "CUDA out of memory"
**Solution**: Either:
- Use CPU: `--device cpu`
- Reduce cases: `--max-cases 5`
- Close other GPU applications

## Output Directory

Results are saved to `test_pipeline_results/gist_ct_GIST_test/`:

```
test_pipeline_results/gist_ct_GIST_test/
├── preproc/
│   ├── scaler.joblib
│   ├── pca.joblib
│   ├── X_train_p.npy
│   └── X_val_p.npy
├── tabpfn_config.json
├── tabpfn_val_predictions.csv
├── tabpfn_metrics.json
└── tabpfn_classification_report.txt
```

## Comparing with Old ROI Pooling

If you want to compare results with the old ROI pooling:

1. Checkout the previous git commit before the pooling changes
2. Run the same test
3. Compare the metrics and feature distributions

The new Average Pooling should give **more stable and consistent** features since:
- No dependency on mask quality
- Consistent pooling across all samples
- Matches the classification head behavior exactly
