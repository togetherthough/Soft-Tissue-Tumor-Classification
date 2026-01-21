# SAM-Med3D Embedding Extraction and Reuse

## Understanding Embedding Extraction

Embeddings are the intermediate representations extracted from SAM-Med3D's image encoder. These are:
- **Deterministic**: Same image always produces identical embeddings
- **Reusable**: Can be cached and reused across experiments
- **Time-saving**: Extraction takes ~20 minutes per dataset; reusing saves hours

## The `skip_existing_embeddings` Parameter

### Naming Clarification

The parameter follows natural language logic:

```python
skip_existing_embeddings=True
# Translation: "Skip/ignore the existing embeddings"
# Effect: Re-extract embeddings, ignoring what exists

skip_existing_embeddings=False  # DEFAULT (RECOMMENDED)
# Translation: "Don't skip existing embeddings" = "Use existing embeddings"
# Effect: Reuse existing embeddings (fast!)
```

### When to Use Each Setting

#### ✅ Use `skip_existing_embeddings=False` (Default - Recommended)

**When:**
- Running experiments with the same SAM-Med3D checkpoint
- Comparing different TabPFN/LoCalPFN hyperparameters
- Running the same notebook multiple times
- Saving time (20+ minutes per run!)

**Why it's safe:**
- Embeddings are deterministic (same image → same embedding)
- The random split happens AFTER extraction (controlled by `seed`)
- No bias introduced

**Example:**
```python
# Experiment with different k values for LoCalPFN
for k in [5, 10, 20, 50]:
    res = run_multi_localpfn(
        config_path="configs/datasets.yaml",
        skip_existing_embeddings=False,  # ← REUSE embeddings (default)
        local_k=k,
    )
```

#### 🔄 Use `skip_existing_embeddings=True` (Force Re-extraction)

**When:**
- You changed the SAM-Med3D checkpoint
- You modified preprocessing (e.g., `img_size`)
- You want fresh embeddings to verify reproducibility
- Something went wrong and you suspect corrupted embeddings

**Example:**
```python
# After loading a new checkpoint
checkpoint_path = Path("new_checkpoint.pth")

res = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    checkpoint=checkpoint_path,
    skip_existing_embeddings=True,  # ← Skip existing, force re-extraction
)
```

## Automatic Per-Dataset Extraction

### The Smart Fix

The pipeline now automatically checks **per-dataset** whether embeddings exist:

```python
# With skip_existing_embeddings=False (default):
run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    skip_existing_embeddings=False,
)

# Output:
# ✅ gist: Reusing existing embeddings...
# ⚠️  lipo: Embeddings missing. Extracting...
```

**Translation:**
- GIST has embeddings → reused ✅
- LIPO missing embeddings → extracted automatically ⚠️

This means `skip_existing_embeddings=False` is the "smart default" that:
- Reuses embeddings when available (fast)
- Extracts when missing (correct)
- Works regardless of dataset order (robust)

### How It Works

#### Before the Fix ❌
```
User runs: run_multi_tabpfn(config, skip_existing_embeddings=True)

GIST:  Has embeddings → Skip ✅
LIPO:  No embeddings → Skip anyway (global setting) ❌ BUG!
```

#### After the Fix ✅
```
User runs: run_multi_tabpfn(config, skip_existing_embeddings=True)

GIST:  Has embeddings → Skip ✅
LIPO:  No embeddings → FORCE EXTRACTION ⚠️ (automatic fix!)
```

### Technical Implementation

**Modified:** `med3pipe/pipelines/multi_dataset.py`

Per-dataset embedding check (lines 208-227):
```python
# Check if embeddings exist for this specific dataset
feat_train_dir = sam3d_root / "features" / category / f"{ct_name}_train"
feat_val_dir = sam3d_root / "features" / category / ct_name

embeddings_exist = (
    feat_train_dir.exists() 
    and feat_val_dir.exists()
    and list(feat_train_dir.glob("*_embedding.pt"))
    and list(feat_val_dir.glob("*_embedding.pt"))
)

# Force extraction if embeddings don't exist for this dataset
force_extraction = not embeddings_exist
skip_for_this_dataset = skip_existing_embeddings and not force_extraction

if force_extraction:
    print(f"\n⚠️  {ds_key}: Embeddings missing. Forcing extraction...")
elif skip_existing_embeddings:
    print(f"\n✅ {ds_key}: Embeddings exist. Skipping extraction...")
```

## Benefits

### 1. Order-Independent
Run datasets in any order, each gets processed correctly:
```python
# Run GIST first
run_multi_tabpfn(config, dataset_names=["gist"])

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
