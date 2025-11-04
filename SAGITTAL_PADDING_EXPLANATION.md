# Why Sagittal Views Look Gray at X=80 (GIST-013)

## Problem Summary
In your orthogonal visualization, the sagittal views (especially at X=80) appear almost completely gray/padded, even though X=80 should contain anatomy.

## Root Cause
The issue is **heavy Z-axis padding**, not missing data in the X dimension.

### Original Volume
- **Shape**: 512×512×**76** 
- Very thin in the Z-axis (only 76 axial slices - typical for some CT scans)

### After Preprocessing to 128³
```
Step 1: ToCanonical()        → 512×512×76
Step 2: ResizeLargestTo(128) → 128×128×19  (scales down uniformly)
Step 3: CropOrPad(128³)      → 128×128×128 (pads Z heavily!)
```

### Padding Distribution
- **X-axis**: NO padding (0 before, 0 after) ✓
- **Y-axis**: NO padding (0 before, 0 after) ✓  
- **Z-axis**: HEAVY padding ⚠️
  - 54 slices of padding BEFORE data
  - 19 slices of real data (Z=54 to Z=72)
  - 55 slices of padding AFTER data

## Why Sagittal Views Look Gray
When viewing a sagittal slice (X=constant, showing Y vs Z plane):
- The Y-axis shows real anatomy across all 128 pixels ✓
- The Z-axis is **85% padding** (109 out of 128 pixels are gray padding) ⚠️
- Only Z=54 to Z=72 contains actual anatomy (15% of the view)
- Result: The slice looks almost entirely gray/uniform

## Verification Results

### At Index Position 80:
```
Axial (Z=80):     PADDING SLICE ⚠️  (Z=80 is outside data region 54-72)
Coronal (Y=80):   DATA SLICE ✓ (Y has no padding)
Sagittal (X=80):  85.2% of view is padding ⚠️ (looks almost all gray!)
```

### Actual Data in Sagittal Slice X=80:
- Total pixels: 128×128 = 16,384
- Data region (Z=54-72): 128×19 = 2,432 pixels (15%)
- Padding region: 128×109 = 13,952 pixels (85%)

## Solutions

### Option 1: Use Better Indices (Quick Fix)
Instead of equal indices `(Z=Y=X)`, use Z values within the data region:

**Current (problematic)**:
```python
indices = [(56, 56, 56), (68, 68, 68), (80, 80, 80)]
# Z=80 is outside data region, sagittal views are 85% padding
```

**Better**:
```python
indices = [(60, 60, 60), (63, 63, 63), (66, 66, 66)]
# All Z values within data region [54-72]
```

### Option 2: Auto-Detect Data Boundaries (Robust Fix)
Modify your visualization function to automatically detect non-padding regions:

```python
def find_data_region(volume, threshold=-900):
    """Find the bounding box of actual data (excluding padding)."""
    mask = volume > threshold
    
    # Find indices where data exists in each dimension
    z_indices = np.where(np.any(mask, axis=(0, 1)))[0]
    y_indices = np.where(np.any(mask, axis=(0, 2)))[0]
    x_indices = np.where(np.any(mask, axis=(1, 2)))[0]
    
    return (
        (x_indices.min(), x_indices.max()),
        (y_indices.min(), y_indices.max()),
        (z_indices.min(), z_indices.max())
    )

# Then select visualization indices from within these ranges
(x_min, x_max), (y_min, y_max), (z_min, z_max) = find_data_region(volume)

# Select 3 evenly-spaced indices within data region
z_indices = np.linspace(z_min, z_max, 3).astype(int)
y_indices = np.linspace(y_min, y_max, 3).astype(int)
x_indices = np.linspace(x_min, x_max, 3).astype(int)
```

### Option 3: Adjust Visualization Window
Crop the displayed Z-axis to only show the data region:

```python
# In your sagittal plotting, only show Z=54 to Z=72
sagittal_slice = volume[x, :, 54:73]  # Only show data region in Z
```

## Summary
- **The preprocessed volume at X=80 DOES have anatomical data** ✓
- **But the sagittal view is 85% padding in the Z dimension** ⚠️
- This is because the original volume is very thin (only 76 slices in Z)
- After resizing and padding to 128³, most of the Z-axis is padding
- **Solution**: Use Z indices within the actual data region (54-72) for visualization

## Files for Reference
- Diagnostic script: `scripts/visualize_x80.py`
- Index mapping: `scripts/trace_index_mapping.py`
- Better visualization: `scripts/create_better_viz.py`
