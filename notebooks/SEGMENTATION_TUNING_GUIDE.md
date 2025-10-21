# SAM Segmentation Tuning Guide

## 🔴 Problem: SAM Showing Red Everywhere

If you see red all over the image (not just on the lesion), SAM is **over-segmenting**.

### Causes:
1. ❌ **Missing or wrong checkpoint** - SAM needs pretrained weights
2. ❌ **Too many prompt points** - confuses SAM
3. ❌ **Low threshold** - accepts weak predictions as positive
4. ❌ **Bad normalization** - SAM gets wrong input

## ✅ Solutions

### Quick Fix: Adjust Settings at Top of Script

```python
# Line 47-49 in SAM_Visualization_Fixed.py

# SAM Segmentation Settings
USE_SINGLE_POINT_PROMPT = True   # Change to True (default)
SEGMENTATION_THRESHOLD = 0.5     # Increase to 0.7 or 0.8
```

### Tuning Guide:

| Problem | Setting | Change To | Effect |
|---------|---------|-----------|--------|
| Red everywhere | `SEGMENTATION_THRESHOLD = 0.5` | `0.7` or `0.8` | Only strong predictions count |
| Too much segmented | `USE_SINGLE_POINT_PROMPT = False` | `True` | More conservative |
| Lesion not fully covered | `SEGMENTATION_THRESHOLD = 0.8` | `0.5` or `0.6` | More liberal |
| Still not working | Check checkpoint | Load proper weights | SAM needs pretrained model |

## 🎯 Recommended Settings

### For Conservative Segmentation (Less False Positives):
```python
USE_SINGLE_POINT_PROMPT = True
SEGMENTATION_THRESHOLD = 0.7  # or even 0.8
```

### For Complete Segmentation (More Coverage):
```python
USE_SINGLE_POINT_PROMPT = False
SEGMENTATION_THRESHOLD = 0.5
```

### Balanced (Default):
```python
USE_SINGLE_POINT_PROMPT = True
SEGMENTATION_THRESHOLD = 0.5
```

## 📊 Console Debug Output

When you run the script, watch for:

```
  Generating SAM segmentation...
    Using SINGLE point prompt at: (64, 45, 23)
    Segmentation stats: min=0.001, max=0.956, threshold=0.5, voxels=12534
```

**Interpreting the stats:**
- `min/max`: Range of predicted probabilities
  - `max > 0.9` → SAM is confident
  - `max < 0.6` → SAM is uncertain (might need checkpoint)
- `threshold`: Current cutoff value
- `voxels`: How many voxels segmented
  - Compare to ground truth voxels

### Example Analysis:

```
Mask sum: 5000 voxels          ← Ground truth size
Segmentation sum: 50000 voxels  ← SAM over-segmented (10x too much!)
```

**Action**: Increase `SEGMENTATION_THRESHOLD` to 0.7 or 0.8

```
Mask sum: 5000 voxels
Segmentation sum: 500 voxels    ← SAM under-segmented
```

**Action**: Decrease `SEGMENTATION_THRESHOLD` to 0.3 or 0.4

## ⚠️ If Nothing Works

### Check 1: Checkpoint Loaded?
Look for this in console:
```
✅ Found checkpoint: .../sam_model_best.pth
📦 Loading checkpoint from: ...
```

If you see:
```
⚠️  No checkpoint found. Loading model without pretrained weights.
```

**Then SAM won't work well!** → Load a proper checkpoint (see CHECKPOINT_LOADING_GUIDE.md)

### Check 2: Normalization Applied?
The code should show:
```
Loading image for SAM (WITH normalization)...
```

If preprocessing is wrong, SAM will fail.

### Check 3: Ground Truth Correct?
Check the green overlay - does it show the actual lesion? If ground truth is wrong, SAM will be wrong.

## 🔧 Advanced: Manual Threshold Per Case

If you want different thresholds for different cases, modify the function:

```python
def generate_sam_with_bbox_prompt(model, image_tensor, gt_mask_tensor, device, 
                                   use_single_point=True, threshold=None):
    # ... existing code ...
    
    # Use custom threshold if provided, otherwise use global
    if threshold is None:
        threshold = SEGMENTATION_THRESHOLD
    
    seg_mask = (seg_prob > threshold).cpu().squeeze().numpy().astype(np.uint8)
```

Then call with:
```python
seg_mask = generate_sam_with_bbox_prompt(
    model, image, gt_mask, device, 
    use_single_point=True,
    threshold=0.75  # Custom threshold for this case
)
```

## 📈 Optimization Strategy

1. **Start conservative**: `USE_SINGLE_POINT_PROMPT=True`, `THRESHOLD=0.7`
2. **Run on sample images**
3. **Check results**:
   - Over-segmenting? → Increase threshold
   - Under-segmenting? → Decrease threshold
   - Still bad? → Check checkpoint, try 3-point prompts
4. **Find sweet spot** for your dataset

## Example: Fixing LIPO Over-Segmentation

Based on your image showing red everywhere:

```python
# Try these settings:
USE_SINGLE_POINT_PROMPT = True   # Single point only
SEGMENTATION_THRESHOLD = 0.75    # Higher threshold

# If still too much:
SEGMENTATION_THRESHOLD = 0.85

# If that makes it too small:
SEGMENTATION_THRESHOLD = 0.65    # Back down a bit
```

**Watch the console output** to see probability ranges and adjust accordingly!

---

**File**: `SAM_Visualization_Fixed.py`  
**Settings Location**: Lines 47-49  
**Function**: Lines 223-291
