# SAM-Med3D Dice Score Computation Guide

## Overview

This guide explains how to compute Dice scores for all images using SAM-Med3D with 11-point iterative refinement.

## What Was Fixed

### 1. **Preprocessing Pipeline Clarification**

The preprocessing pipeline uses `ResizeLargestTo` + `CropOrPad`, which is the **CORRECT** approach:

```python
transform = tio.Compose([
    tio.ToCanonical(),                                    # Standard RAS+ orientation
    ResizeLargestTo(target_size=128),                    # Resize largest dim → 128 (preserves all data)
    tio.CropOrPad(target_shape=(128, 128, 128)),         # MINIMAL padding (only after resize)
    tio.ZNormalization(masking_method=_znorm_masking_method),  # CRITICAL for SAM!
])
```

**Why this is correct:**
- `ResizeLargestTo(128)` scales the largest dimension to 128 while preserving aspect ratio
  - Example: (200, 150, 100) → (128, 96, 64) [ALL data preserved!]
- `CropOrPad` only does **minimal padding** after the resize
  - Example: (128, 96, 64) → (128, 128, 128) [only pads smaller dims]
- This minimizes data loss compared to direct `CropOrPad` without resize

See `docs/PREPROCESSING_UPGRADE.md` for full details.

### 2. **Dataset Discovery**

Fixed the `find_all_datasets()` function to properly discover ALL available datasets in the SAM-Med3D directory structure:

- Searches `SAM-Med3D-main/data/train/` and `SAM-Med3D-main/data/validation/`
- Finds all category directories (gist, lipo, crlm, desmoid, liver, melanoma)
- Matches images to labels automatically
- Returns structured dictionary with all dataset information

Currently available datasets:
- **gist** (train + validation)
- **lipo** (train + validation)

### 3. **11-Click Iterative Refinement**

The notebook and script now use proper iterative refinement matching the SAM-Med3D training procedure:

1. **First click**: Center of ground truth mask
2. **Clicks 2-11**: Sample from error regions
   - False negatives (missed tumor regions) → positive clicks
   - False positives (over-segmented regions) → negative clicks
3. Each iteration refines the segmentation progressively

This matches the SAM-Med3D paper and achieves much higher Dice scores than single-shot approaches.

## Usage

### Option 1: Run the Standalone Script

```bash
# From project root
python scripts/compute_sam_dice_scores.py
```

This will:
- Load the SAM-Med3D model
- Find all available datasets
- Compute Dice scores for all images with 11-click refinement
- Save results to `results/sam_dice_scores_11clicks.csv`

### Option 2: Use the Notebook

Open `notebooks/visualization/SAM_Visualization.ipynb` and run the cells in the **"Compute Dice Scores for ALL Images"** section:

1. **Discover datasets**: Cell 39
2. **Define computation function**: Cell 40
3. **Run computation**: Cell 41
4. **Save results**: Cell 42

## Configuration

You can adjust these parameters at the top of the notebook or script:

```python
NUM_CLICKS = 11                    # Number of iterative refinement clicks (1-20)
SEGMENTATION_THRESHOLD = 0.5       # Threshold for final segmentation mask (0.0-1.0)
img_size = 128                     # Preprocessing size (must be 128 for SAM-Med3D)
```

## Output Format

The results are saved as a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name (gist, lipo, etc.) |
| `split` | Train or validation |
| `case` | Case identifier (e.g., GIST-001_CT) |
| `dice_score` | Dice coefficient (0.0-1.0) |
| `num_clicks` | Number of clicks used (11) |
| `threshold` | Segmentation threshold (0.5) |
| `gt_voxels` | Number of voxels in ground truth mask |
| `pred_voxels` | Number of voxels in predicted mask |
| `error` | Error message (if computation failed) |

## Expected Results

With 11-click iterative refinement, you should expect:

- **Mean Dice > 0.80** for most tumor types
- Progressive improvement with each click
- Much better results than single-shot approaches

## Files

### New/Updated Files

1. **`scripts/compute_sam_dice_scores.py`** - Standalone script for batch computation
2. **`notebooks/visualization/SAM_Visualization.ipynb`** - Updated notebook with dice computation cells
3. **`SAM_DICE_SCORES_GUIDE.md`** - This guide

### Key Functions

- `find_all_datasets()` - Discovers all available datasets
- `load_volume_for_sam()` - Loads and preprocesses image for SAM
- `load_mask_for_sam()` - Loads and preprocesses mask
- `generate_sam_with_iterative_refinement()` - Runs SAM with 11-click refinement
- `compute_dice_score()` - Computes Dice coefficient
- `compute_all_dice_scores()` - Batch processes all images

## Troubleshooting

### No datasets found
- Check that data exists in `SAM-Med3D-main/SAM-Med3D-main/data/train/` or `.../data/validation/`
- Directory structure should be: `data/train/{category}/ct_{CATEGORY}/imagesTr/` and `labelsTr/`

### Out of memory errors
- Use GPU if available (much faster)
- Reduce number of images processed at once
- Consider processing datasets one at a time

### Low Dice scores
- Check that Z-normalization is enabled (CRITICAL for SAM!)
- Verify ground truth masks are correct
- Try adjusting the segmentation threshold (0.3-0.7 range)

## References

- **Preprocessing docs**: `docs/PREPROCESSING_UPGRADE.md`
- **SAM-Med3D paper**: https://arxiv.org/abs/2310.15161
- **Padding explanation**: `SAGITTAL_PADDING_EXPLANATION.md`
