# SAM-Med3D Dice Score Debugging Summary

## Problem Statement
SAM-Med3D with 11-click iterative refinement produces extremely low Dice scores (mean ≈0.035) across all datasets. Median Dice is 0.0 for most datasets, indicating systematic failure.

## Completed Diagnostic Steps

### 1. Dataset Discovery Fix ✅
**Issue**: Script was only finding datasets in SAM-Med3D repo structure (train/validation splits).  
**Fix**: Rewrote `discover_cases_from_config()` to read from `configs/datasets_analysis.yaml` and process raw data at `data/<dataset>/<case>/1/NIFTI/`.  
**Result**: Now processes all 6 datasets (crlm, desmoid, gist, lipo, liver, melanoma) = 930 total cases.

### 2. Preprocessing Validation ✅
**Tool**: Created `scripts/debug_sam_preprocessing.py`  
**Findings**:
- Raw images and masks load correctly
- `ResizeLargestTo(128) → CropOrPad(128³) → ZNormalization` pipeline behaves as expected
- Voxel counts reduce proportionally (e.g., CRLM mask: 14,766 → 242 voxels)
- Modality-aware clamping now works (CT: [-1000,1000], MR: no clamp)

**Conclusion**: Preprocessing is **NOT** the issue.

### 3. Iterative Refinement Deep-Dive ✅
**Tool**: Added `--debug-case` flag to log per-click metrics  
**Test Case**: `liver:Liver-001_MR` (representative failing case)  
**Debug Log**: `results/debug_logs/liver_Liver-001_MR_debug.json`

#### Key Findings from Debug Log

| Metric | Value | Analysis |
|--------|-------|----------|
| **GT voxels** | 639 | Very small lesion after preprocessing |
| **Pred voxels** | 7,782 | 12× larger than GT → massive false positive region |
| **Per-click Dice** | 0.0006 (flat) | Zero improvement across 11 clicks |
| **Final prob stats** | mean=0.005, max=0.995 | Model produces confident predictions but in **wrong locations** |
| **Click pattern** | Mixed pos/neg | Refinement loop samples errors but model doesn't converge |

#### Per-Click Evolution
```
Click  Dice    Voxels  Label
1      0.0006  1,291   pos (GT center)
2      0.0006  1,180   neg (false positive)
3      0.0006  1,238   pos (false negative)
...
11     0.0006  7,782   pos (false negative)
```

**Critical Observation**: Dice **never improves**. Predicted mask grows steadily but remains spatially misaligned with GT.

## Root Cause Analysis

### Primary Issue: Spatial Misalignment
The model consistently predicts high-confidence masks in the **wrong anatomical locations**:
1. First click at GT center produces ~1,300 voxels (2× GT size) with only ~2 voxels overlapping
2. Error-based clicks sample from misaligned regions, reinforcing incorrect predictions
3. Model's image embeddings may not be suitable for these small, downsampled lesions (639 voxels in 128³ space)

### Contributing Factors

1. **Extreme Downsampling**
   - Original lesions (e.g., 14k+ voxels) → 200-600 voxels after resize to 128³
   - Small targets become difficult for SAM's encoder to localize accurately

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

#### Option 1: Multi-Point Initialization
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

## Next Steps

### Immediate Action Items
1. **Implement Option 1 (Multi-Point Init)** - Fastest fix with high likelihood of improvement
2. **Re-run debug on 3-5 cases per dataset** to validate improvement
3. **Compare before/after Dice distributions**

### Code Changes Required
1. Modify `generate_sam_with_iterative_refinement()`:
   - Accept `num_init_points` parameter (default=5)
   - Sample multiple GT points for first click instead of just center
   - Concatenate all initial points before first forward pass

2. Update `sample_next_click()`:
   - Add bias parameter to favor false negatives early
   - Return multiple points if needed

3. Add visualization:
   - Save PNG overlays of pred vs GT for debug cases
   - Create notebook cell to inspect failed cases interactively

## Tools Created for Iteration

### Fast Subset Testing
```bash
# Test single dataset with 3 cases
python scripts/compute_sam_dice_scores.py --datasets liver --max-cases 3

# Debug specific case with full logging
python scripts/compute_sam_dice_scores.py --datasets liver --max-cases 3 \
  --debug-case liver:Liver-001_MR --debug-dir results/debug_logs
```

### Files Modified
- `scripts/compute_sam_dice_scores.py`: Added CLI filtering, debug logging, per-click metrics
- `scripts/debug_sam_preprocessing.py`: Preprocessing validation tool

### Debug Outputs
- `results/debug_logs/<dataset>_<case>_debug.json`: Per-click Dice, points, labels, prob stats
- Console logs show click-by-click progress with verbose mode

## Summary

**Preprocessing is correct**. The issue is **prompt strategy + spatial misalignment**. The model produces confident predictions but in the wrong locations because:
1. Single center-point prompt is insufficient for tiny downsampled lesions
2. Error-based refinement reinforces misalignment instead of correcting it
3. No recovery mechanism when iterative loop diverges

**Recommended first fix**: Implement multi-point initialization (Option 1) and re-evaluate.
