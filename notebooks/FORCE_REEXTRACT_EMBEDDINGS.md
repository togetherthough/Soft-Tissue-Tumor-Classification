# Force Re-extraction of SAM-Med3D Embeddings

## Problem

Old cached embeddings were extracted with **random weights** (before loading pretrained checkpoint).
The pipeline was reusing these garbage embeddings, preventing you from seeing improvements with pretrained weights.

## Solution

Added `skip_existing_embeddings` parameter to force re-extraction.

## Usage

### Option 1: Delete Old Features (Already Done ✅)

Old features have been deleted from:
- `SAM-Med3D-main/features/gist/`
- `SAM-Med3D-main/features/lipo/`

Just rerun your notebook - it will extract fresh embeddings with pretrained weights.

### Option 2: Force Re-extraction Programmatically

In your `Experiment1-Benchmarks.ipynb`, add this parameter:

```python
# TabPFN
res_tab = run_multi_tabpfn(
    config_path=config_path,
    dataset_names=datasets_to_run,
    outputs_base_dir=outputs_base,
    checkpoint=checkpoint_path,
    skip_existing_embeddings=False,  # ← Force re-extraction
)

# LoCalPFN
res_loc = run_multi_localpfn(
    config_path=config_path,
    dataset_names=datasets_to_run,
    outputs_base_dir=outputs_base,
    checkpoint=checkpoint_path,
    skip_existing_embeddings=False,  # ← Force re-extraction
    local_k=8, local_fit_adapter=True, ...
)
```

## Why This Matters

- **Random weights** = meaningless embeddings (even if consistent with same seed)
- **Pretrained weights** = learned medical image representations from 130k+ images
- Re-extracting with pretrained weights should give you **much better** TabPFN/LoCalPFN performance

## Default Behavior

- `skip_existing_embeddings=True` (default): Uses cached embeddings if they exist (faster, but uses old features)
- `skip_existing_embeddings=False`: Always re-extracts embeddings (slower, but ensures fresh features)

## Summary of Changes

Modified files:
1. `med3pipe/sam/core.py` - Added `skip_existing` parameter to `extract_embeddings_train_val()`
2. `med3pipe/pipelines/end_to_end.py` - Added `skip_existing_embeddings` parameter to `run_single_dataset()`
3. `med3pipe/pipelines/multi_dataset.py` - Added `skip_existing_embeddings` to all multi-dataset functions

Now rerun your experiments and you should see **significant improvements** with pretrained weights! 🎯
