# Embedding Reuse Guide

## Understanding `skip_existing_embeddings`

**TL;DR:**
- ✅ `skip_existing_embeddings=False` → **Reuse embeddings** (fast, recommended, DEFAULT)
- 🔄 `skip_existing_embeddings=True` → **Skip/ignore existing, re-extract** (slow, useful when changing checkpoint)

---

## What It Actually Means

The parameter name follows natural language interpretation:

```python
skip_existing_embeddings=True
# Translation: "Skip/ignore the existing embeddings"
# Effect: Re-extract embeddings, ignoring what exists

skip_existing_embeddings=False  # DEFAULT
# Translation: "Don't skip existing embeddings" = "Use existing embeddings"
# Effect: Reuse existing embeddings (fast!)
```

### The Interpretation

- `skip_existing_embeddings` = should I skip/ignore existing embeddings?
  - `True` → Yes, skip them → re-extract
  - `False` → No, don't skip them → reuse them

---

## When to Use Each Setting

### ✅ Use `skip_existing_embeddings=False` (Default - Recommended)

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

---

### 🔄 Use `skip_existing_embeddings=True` (Force Re-extraction)

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
    skip_existing_embeddings=True,  # ← Skip existing, force re-extraction with new checkpoint
)
```

---

## The Automatic Safety Net

**Good news:** With the latest fix, the pipeline is smart about missing embeddings!

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

---

## Quick Reference

| Goal | Setting | Time | Notes |
|------|---------|------|-------|
| **Reuse embeddings** | `False` | Fast ⚡ | Recommended for experiments (DEFAULT) |
| **Force re-extract** | `True` | Slow 🐌 | Use after changing checkpoint |
| **Auto-handle missing** | `False` | Smart 🧠 | Default behavior (best of both!) |

---

## Common Scenarios

### Scenario 1: Running Notebook for First Time
```python
# First run - embeddings don't exist yet
res = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    skip_existing_embeddings=False,  # Will auto-extract (nothing to reuse)
)
```

### Scenario 2: Running Notebook Again
```python
# Second run - embeddings exist
res = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    skip_existing_embeddings=False,  # Will reuse (fast!)
)
```

### Scenario 3: Testing Different Hyperparameters
```python
# Try different TabPFN ensemble sizes
for n_ensemble in [8, 16, 32]:
    res = run_multi_tabpfn(
        config_path="configs/datasets.yaml",
        skip_existing_embeddings=False,  # Reuse embeddings each time (default)
        tabpfn_clf_kwargs={'N_ensemble_configurations': n_ensemble},
    )
```

### Scenario 4: New Checkpoint Loaded
```python
# After downloading a new SAM-Med3D checkpoint
res = run_multi_tabpfn(
    config_path="configs/datasets.yaml",
    checkpoint=Path("new_checkpoint.pth"),
    skip_existing_embeddings=True,  # Skip existing, force re-extraction
)
```

---

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

---

## Time Savings

Typical embedding extraction times:

| Dataset | Images | Time per Run | Savings with Reuse |
|---------|--------|-------------|-------------------|
| GIST | 246 | ~15 min | ⚡ Skip |
| LIPO | 115 | ~8 min | ⚡ Skip |
| **Total** | 361 | **~23 min** | **Saved!** |

**Over 10 experimental runs:** ~4 hours saved! 🎉

---

## Troubleshooting

### Problem: "Embeddings missing" even though they should exist

**Check:**
```python
from pathlib import Path

sam3d_root = Path("SAM-Med3D-main/SAM-Med3D-main")
gist_train = sam3d_root / "features/gist/ct_GIST_train"
gist_val = sam3d_root / "features/gist/ct_GIST"

print(f"Train exists: {gist_train.exists()}")
print(f"Train files: {len(list(gist_train.glob('*_embedding.pt')))}")
print(f"Val exists: {gist_val.exists()}")
print(f"Val files: {len(list(gist_val.glob('*_embedding.pt')))}")
```

**Solution:** Run with `skip_existing_embeddings=False` once to extract.

---

### Problem: Results differ between runs

**Possible causes:**
1. ❌ Different `seed` in config
2. ❌ Different `split_ratio` in config
3. ❌ Added/removed samples
4. ❌ Changed SAM-Med3D checkpoint between runs

**Solution:** Ensure your `configs/datasets.yaml` is unchanged.

---

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

---

## Better Names We Wish We Had

If we could rename this parameter without breaking existing code:

```python
# Option 1: Positive framing
reuse_embeddings=True        # Clear intention
reuse_embeddings=False       # Clear intention

# Option 2: Action-focused  
force_reextract=False        # Don't force → reuse
force_reextract=True         # Force re-extraction

# Option 3: Explicit
embedding_mode="reuse"       # Reuse if exist
embedding_mode="force"       # Always extract
embedding_mode="auto"        # Smart default (current behavior)
```

The parameter now follows natural language:
- `skip_existing_embeddings=False` → **DON'T SKIP** = **REUSE** (what you usually want, DEFAULT)
- `skip_existing_embeddings=True` → **SKIP EXISTING** = **RE-EXTRACT** (only when needed)

---

**Questions? Check the updated docstrings in the code or see `EMBEDDING_EXTRACTION_FIX.md`**
