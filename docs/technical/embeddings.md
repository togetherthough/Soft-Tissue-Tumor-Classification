# SAM-Med3D Embedding Extraction

> Guide to extracting, caching, and reusing SAM-Med3D embeddings

## Overview

Embeddings are intermediate representations extracted from SAM-Med3D's image encoder. They are:

- **Deterministic**: Same input always produces identical output
- **Reusable**: Can be cached and reused across experiments
- **Time-saving**: Extraction takes ~20 minutes per dataset; reusing saves hours

---

## Key Parameter: `skip_existing_embeddings`

This parameter controls whether to use cached embeddings or re-extract them.

| Value | Behavior | Use Case |
|-------|----------|----------|
| `False` (default) | Reuse existing embeddings | Typical experiments |
| `True` | Skip existing, force re-extraction | Changed checkpoint or preprocessing |

### When to Use Each Setting

#### `skip_existing_embeddings=False` (Recommended)

Use when:
- Running experiments with the same SAM-Med3D checkpoint
- Comparing different TabPFN/LoCalPFN hyperparameters
- Running experiments multiple times
- Saving time (20+ minutes per dataset)

```python
res = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    skip_existing_embeddings=False,  # Reuse embeddings
)
```

#### `skip_existing_embeddings=True`

Use when:
- Changed the SAM-Med3D checkpoint
- Modified preprocessing (e.g., `img_size`)
- Need to verify reproducibility
- Suspect corrupted embeddings

```python
res = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    skip_existing_embeddings=True,  # Force re-extraction
)
```

---

## Automatic Per-Dataset Handling

The pipeline automatically checks each dataset:

```
run_multi_tabpfn(config, skip_existing_embeddings=False)

Output:
✅ gist: Reusing existing embeddings...
⚠️  lipo: Embeddings missing. Extracting...
```

- Datasets with cached embeddings → reused
- Datasets without embeddings → extracted automatically

---

## Embedding Location

Embeddings are stored in the SAM-Med3D feature directory:

```
sam-med3d/
└── features/
    └── <category>/
        ├── <ct_name>_train/
        │   ├── CASE-001_embedding.pt
        │   ├── CASE-002_embedding.pt
        │   └── ...
        └── <ct_name>/  # validation
            ├── CASE-003_embedding.pt
            └── ...
```

---

## Manual Extraction

### Using the Pipeline

```python
from med3pipe import (
    build_sam3d_model,
    extract_embeddings_train_val,
    Sam3DPaths,
)

# Build model
model = build_sam3d_model(
    sam3d_root="sam-med3d",
    checkpoint="sam-med3d/ckpt/sam_med3d_turbo.pth",
)

# Extract embeddings
paths = Sam3DPaths(
    sam3d_root="sam-med3d",
    category="gist",
    ct_name="ct_GIST",
)
extract_embeddings_train_val(paths, model, sam3d_root="sam-med3d")
```

### Loading Embeddings

```python
from med3pipe import load_pooled_features

X_train, ids_train = load_pooled_features(
    "sam-med3d/features/gist/ct_GIST_train",
    paths.labels_tr,
)
print(f"Shape: {X_train.shape}")  # (n_samples, 384)
```

---

## Pooling Strategies

Embeddings are 3D feature maps (C×D×H×W). Pooling converts them to fixed-length vectors.

| Strategy | Description | Output Dim |
|----------|-------------|------------|
| **Global Average** | Mean across spatial dims | C |
| **Multiscale** | Multiple pool sizes | C × scales |
| **Percentile** | Percentile-based pooling | C × percentiles |

Default: Percentile Pooling (1,920-dim vectors)

---

## Caching Behavior

### What's Cached

- Embedding tensors (`.pt` files)
- One file per case

### What's NOT Cached

- Pooled feature vectors (recomputed each run)
- PCA transformations
- Classification results

### Cache Invalidation

Delete cached embeddings when:
1. Checkpoint changed
2. Preprocessing changed
3. Corrupted files detected

```bash
# Remove all embeddings for a dataset
rm -rf sam-med3d/features/gist/
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Embeddings missing" | Run with `skip_existing_embeddings=False` (auto-extracts) |
| "Dimension mismatch" | Re-extract with `skip_existing_embeddings=True` |
| "Corrupted embeddings" | Delete cache and re-extract |

---

## Related Documentation

- [Preprocessing](preprocessing.md) — Data preprocessing pipeline
- [Multi-Dataset](multi-dataset.md) — Processing multiple datasets
- [Quick Reference](../QUICK_REFERENCE.md) — Command cheat sheet

# Run LIPO later - embeddings automatically extracted!
run_multi_tabpfn(config, dataset_names=["lipo"])
```

### 2. Works for Both Methods
The fix applies to **both** TabPFN and LoCalPFN:
```python
# TabPFN - automatically extracts missing embeddings
run_multi_tabpfn(config)

# LoCalPFN - also automatically extracts missing embeddings
run_multi_localpfn(config)
```

### 3. Smart Caching
- Datasets with embeddings → skipped (fast)
- Datasets without embeddings → extracted (automatic)
- No manual intervention needed

### 4. Clear Feedback
```
⚠️  lipo: Embeddings missing. Forcing extraction...
✅ gist: Embeddings exist. Skipping extraction...
```

## Usage Examples

### Run All Datasets (Automatic Extraction)
```python
from med3pipe.pipelines import run_multi_tabpfn

# Automatically extracts embeddings for any dataset that needs them
result = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    skip_existing_embeddings=False,  # Smart default (recommended)
)
```

### CLI Usage
```bash
# TabPFN - automatically handles missing embeddings
python -m med3pipe multi-tabpfn --config configs/datasets.yaml

# LoCalPFN - also automatically handles missing embeddings
python -m med3pipe multi-localpfn --config configs/datasets.yaml --local-k 50
```

## Time Savings

Typical embedding extraction times:

| Dataset | Images | Time per Run | Savings with Reuse |
|---------|--------|-------------|-------------------|
| GIST | 246 | ~15 min | ⚡ Skip |
| LIPO | 115 | ~8 min | ⚡ Skip |
| **Total** | 361 | **~23 min** | **Saved!** |

**Over 10 experimental runs:** ~4 hours saved! 🎉

## Does Reusing Embeddings Introduce Bias?

**No!** Here's why:

### Embedding Extraction (No Randomness)
```
Image → SAM-Med3D → Embedding
```
- Same image always produces identical embedding
- Deterministic process
- Like taking a photograph

### Train/Val Split (Randomness Controlled)
```
All embeddings → Stratified split (seed=2025) → Train/Val
```
- Split happens AFTER embeddings are created
- Controlled by `seed` in your config
- Same seed = same split

### Verification
```yaml
# configs/datasets.yaml
datasets:
  gist:
    split:
      ratio: 0.8
      seed: 2025  # ← This controls reproducibility
```

**Result:** Reusing embeddings = reusing a cached file. No bias introduced! ✅

## Troubleshooting

### Problem: "Embeddings missing" even though they should exist

**Check:**
```python
from pathlib import Path

sam3d_root = Path("sam-med3d")
gist_train = sam3d_root / "features/gist/ct_GIST_train"
gist_val = sam3d_root / "features/gist/ct_GIST"

print(f"Train exists: {gist_train.exists()}")
print(f"Train files: {len(list(gist_train.glob('*_embedding.pt')))}")
print(f"Val exists: {gist_val.exists()}")
print(f"Val files: {len(list(gist_val.glob('*_embedding.pt')))}")
```

**Solution:** Run with `skip_existing_embeddings=False` once to extract.

### Problem: Results differ between runs

**Possible causes:**
1. ❌ Different `seed` in config
2. ❌ Different `split_ratio` in config
3. ❌ Added/removed samples
4. ❌ Changed SAM-Med3D checkpoint between runs

**Solution:** Ensure your `configs/datasets.yaml` is unchanged.

## Quick Reference

| Goal | Setting | Time | Notes |
|------|---------|------|-------|
| **Reuse embeddings** | `False` | Fast ⚡ | Recommended for experiments (DEFAULT) |
| **Force re-extract** | `True` | Slow 🐌 | Use after changing checkpoint |
| **Auto-handle missing** | `False` | Smart 🧠 | Default behavior (best of both!) |

## Summary

**For 99% of use cases:**
```python
skip_existing_embeddings=False  # ← Use this! (DEFAULT)
```

This setting:
- ✅ Reuses embeddings when available (fast)
- ✅ Extracts when missing (correct)
- ✅ No bias introduced (reproducible)
- ✅ Saves hours of computation time

**Only use `True` when:**
- 🔄 You changed the SAM-Med3D checkpoint
- 🔄 You want fresh embeddings to verify reproducibility

## Files Modified

- `med3pipe/pipelines/multi_dataset.py` - Per-dataset embedding checks
  - Function: `_run_multi_core()` (lines 208-244)
  - Functions using it: `run_multi_tabpfn()`, `run_multi_localpfn()`

**Date:** October 21, 2025  
**Affects:** `run_multi_tabpfn()`, `run_multi_localpfn()`, and all multi-dataset workflows
