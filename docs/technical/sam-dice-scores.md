# SAM-Med3D Dice Score Evaluation

## Overview

This guide explains how to compute and debug Dice scores for SAM-Med3D segmentations using 11-point iterative refinement.

## Quick Start

### Run Dice Score Computation

```bash
# From project root - compute for all datasets
python scripts/compute_sam_dice_scores.py

# Compute for specific datasets only
python scripts/compute_sam_dice_scores.py --datasets liver gist

# Limit number of cases per dataset (for testing)
python scripts/compute_sam_dice_scores.py --datasets liver --max-cases 3

# Debug specific case with full logging
python scripts/compute_sam_dice_scores.py \
    --datasets liver \
    --max-cases 3 \
    --debug-case liver:Liver-001_MR \
    --debug-dir results/debug_logs
```

### Output

Results are saved to `results/sam_dice_scores_11clicks.csv`:

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

## Preprocessing Pipeline

The preprocessing uses `ResizeLargestTo` + `CropOrPad` (CORRECT approach):

```python
transform = tio.Compose([
    tio.ToCanonical(),                                    # Standard RAS+ orientation
    ResizeLargestTo(target_size=128),                    # Resize largest dim → 128
    tio.CropOrPad(target_shape=(128, 128, 128)),         # Minimal padding
    tio.ZNormalization(masking_method=_znorm_masking_method),  # CRITICAL for SAM!
])
```

**Why this is correct:**
- `ResizeLargestTo(128)` scales the largest dimension to 128 while preserving aspect ratio
  - Example: (200, 150, 100) → (128, 96, 64) [ALL data preserved!]
- `CropOrPad` only does **minimal padding** after the resize
  - Example: (128, 96, 64) → (128, 128, 128) [only pads smaller dims]
- This minimizes data loss compared to direct `CropOrPad` without resize

See `docs/technical/preprocessing.md` (formerly `PREPROCESSING_UPGRADE.md`) for full details.

## 11-Click Iterative Refinement

The evaluation uses proper iterative refinement matching SAM-Med3D training:

1. **First click**: Center of ground truth mask (positive)
2. **Clicks 2-11**: Sample from error regions
   - False negatives (missed tumor regions) → positive clicks
   - False positives (over-segmented regions) → negative clicks
3. Each iteration refines the segmentation progressively

### Expected Results

With 11-click iterative refinement:
- **Mean Dice > 0.80** for most tumor types (ideal case)
- Progressive improvement with each click
- Much better than single-shot approaches

**Note:** Current results show mean Dice ≈0.035, indicating systematic issues (see Debugging section).

## Configuration

Adjust these parameters in the script or notebook:

```python
NUM_CLICKS = 11                    # Number of iterative refinement clicks (1-20)
SEGMENTATION_THRESHOLD = 0.5       # Threshold for final segmentation mask (0.0-1.0)
img_size = 128                     # Preprocessing size (must be 128 for SAM-Med3D)
```

## Debugging Low Dice Scores

### Problem Statement

SAM-Med3D with 11-click iterative refinement may produce extremely low Dice scores (mean ≈0.035) across all datasets, with median Dice of 0.0 for most datasets, indicating systematic failure.

### Completed Diagnostic Steps

#### 1. Dataset Discovery ✅
**Fixed:** Script now reads from `configs/datasets_analysis.yaml` and processes raw data at `data/<dataset>/<case>/1/NIFTI/`.  
**Result:** Processes all 6 datasets (crlm, desmoid, gist, lipo, liver, melanoma) = 930 total cases.

#### 2. Preprocessing Validation ✅
**Tool:** `scripts/debug_sam_preprocessing.py`  
**Findings:**
- Raw images and masks load correctly
- Preprocessing pipeline behaves as expected
- Voxel counts reduce proportionally (e.g., CRLM mask: 14,766 → 242 voxels)
- Modality-aware clamping works (CT: [-1000,1000], MR: no clamp)

**Conclusion:** Preprocessing is **NOT** the issue.

#### 3. Iterative Refinement Analysis ✅
**Tool:** Added `--debug-case` flag for per-click metrics  
**Debug Output:** `results/debug_logs/{dataset}_{case}_debug.json`

**Key Findings:**

| Metric | Typical Value | Analysis |
|--------|---------------|----------|
| **GT voxels** | 200-700 | Very small lesions after preprocessing |
| **Pred voxels** | 7,000+ | 10-20× larger than GT → massive false positive region |
| **Per-click Dice** | ~0.001 (flat) | Zero improvement across 11 clicks |
| **Final prob stats** | mean=0.005, max=0.995 | Model is confident but **spatially misaligned** |
| **Click pattern** | Mixed pos/neg | Refinement samples errors but model doesn't converge |

**Critical Observation:** Dice **never improves**. Predicted mask grows but remains spatially misaligned with GT.

### Root Cause Analysis

#### Primary Issue: Spatial Misalignment
The model consistently predicts high-confidence masks in the **wrong anatomical locations**:
1. First click at GT center produces large prediction with minimal overlap
2. Error-based clicks sample from misaligned regions, reinforcing incorrect predictions
3. Model's image embeddings may not be suitable for small, downsampled lesions

#### Contributing Factors

1. **Extreme Downsampling**
   - Original lesions (14k+ voxels) → 200-600 voxels after resize to 128³
   - Small targets become difficult for SAM's encoder to localize

2. **Prompt Initialization**
   - Single center-point click may be insufficient for tiny, sparse masks
   - SAM-Med3D was likely trained on larger/denser structures

3. **Error Accumulation**
   - `sample_next_click()` samples from false positives/negatives
   - If initial mask is severely misaligned, subsequent clicks reinforce errors
   - No "reset" mechanism when Dice stops improving

4. **Threshold Not the Issue**
   - Max probability ≈0.995 shows model is confident
   - Thresholding at 0.5 is appropriate; problem is spatial, not confidence-based

## Recommendations (Prioritized)

### High Priority: Improve Prompt Strategy

#### Option 1: Multi-Point Initialization (Recommended First Fix)
Instead of single center click, use **multiple seed points**:
```python
# Sample 3-5 points from GT region
coords = np.argwhere(gt_mask > 0)
if len(coords) > 5:
    indices = np.random.choice(len(coords), 5, replace=False)
    seed_points = coords[indices]
else:
    seed_points = coords
```

#### Option 2: Bounding Box Prompt
Use GT bounding box instead of/in addition to point prompts:
```python
coords = np.argwhere(gt_mask > 0)
z_min, y_min, x_min = coords.min(axis=0)
z_max, y_max, x_max = coords.max(axis=0)
bbox = [x_min, y_min, z_min, x_max, y_max, z_max]
# Pass bbox to model.prompt_encoder
```

#### Option 3: Adaptive Click Sampling
Modify `sample_next_click()` to:
- Bias heavily toward false negatives early (clicks 1-5)
- Only sample false positives if Dice is improving
- Add random jitter to avoid getting stuck in local minima

### Medium Priority: Preprocessing Adjustments

#### Option 4: Less Aggressive Downsampling
For datasets with tiny lesions, use larger target size:
```python
# Dynamic sizing based on original volume
original_max_dim = max(image_sitk.GetSize())
if original_max_dim > 256:
    img_size = 192  # Use 192³ instead of 128³
```

#### Option 5: GT Dilation
Slightly dilate ground truth masks before computing clicks:
```python
from scipy.ndimage import binary_dilation
gt_mask_dilated = binary_dilation(gt_mask, iterations=2)
# Use dilated mask for click sampling only
```

### Low Priority: Model/Training

#### Option 6: Fine-tune SAM-Med3D
- Fine-tune on your specific datasets with smaller lesions
- Train with multi-point prompts from the start

#### Option 7: Ensemble/Fallback
- If Dice doesn't improve after 5 clicks, restart with different seed points
- Average predictions from multiple prompt strategies

## Tools Created

### Fast Subset Testing
```bash
# Test single dataset with 3 cases
python scripts/compute_sam_dice_scores.py --datasets liver --max-cases 3

# Debug specific case with full logging
python scripts/compute_sam_dice_scores.py \
    --datasets liver \
    --max-cases 3 \
    --debug-case liver:Liver-001_MR \
    --debug-dir results/debug_logs
```

### Files Modified
- `scripts/compute_sam_dice_scores.py` - Added CLI filtering, debug logging, per-click metrics
- `scripts/debug_sam_preprocessing.py` - Preprocessing validation tool

### Debug Outputs
- `results/debug_logs/<dataset>_<case>_debug.json` - Per-click Dice, points, labels, prob stats
- Console logs show click-by-click progress with verbose mode

## Troubleshooting

### No datasets found
- Check that data exists in `data/<dataset>/<case>/1/NIFTI/`
- Verify `configs/datasets_analysis.yaml` is correctly configured

### Out of memory errors
- Use GPU if available (much faster)
- Reduce number of cases: `--max-cases 10`
- Consider processing datasets one at a time

### Low Dice scores
- Check that Z-normalization is enabled (CRITICAL for SAM!)
- Verify ground truth masks are correct
- Try adjusting the segmentation threshold (0.3-0.7 range)
- Implement multi-point initialization (see Recommendations)

## Files and Functions

### Key Scripts
1. **`scripts/compute_sam_dice_scores.py`** - Batch Dice computation with debug options
2. **`scripts/debug_sam_preprocessing.py`** - Preprocessing validation tool
3. **`notebooks/visualization/SAM_Visualization.ipynb`** - Interactive notebook

### Key Functions
- `find_all_datasets()` - Discovers available datasets
- `load_volume_for_sam()` - Loads and preprocesses image
- `load_mask_for_sam()` - Loads and preprocesses mask
- `generate_sam_with_iterative_refinement()` - Runs SAM with 11-click refinement
- `compute_dice_score()` - Computes Dice coefficient
- `compute_all_dice_scores()` - Batch processes all images

## Summary

**Preprocessing is correct**. The issue is **prompt strategy + spatial misalignment**:
1. Single center-point prompt is insufficient for tiny downsampled lesions
2. Error-based refinement reinforces misalignment instead of correcting it
3. No recovery mechanism when iterative loop diverges

**Recommended first fix:** Implement multi-point initialization (Option 1) and re-evaluate.

## References

- **Preprocessing docs**: `docs/technical/preprocessing.md`
- **SAM-Med3D paper**: https://arxiv.org/abs/2310.15161
- **Dataset analysis**: `configs/datasets_analysis.yaml`
