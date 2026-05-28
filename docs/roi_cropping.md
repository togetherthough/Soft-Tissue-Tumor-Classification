# ROI-Centric Preprocessing Approach

## Overview
This document describes the **ROI-centric (Region of Interest) preprocessing** approach added to the Med3Tab-PFN pipeline as an alternative to the default full-volume preprocessing.

## Motivation

### Problem with Full-Volume Approach
The original `prepare_for_sam3d()` function resizes entire CT scans (e.g., 512×512×300) to 128×128×128. This causes:

1. **Resolution Loss**: Small tumors (e.g., 2cm GIST) shrink from ~40 voxels to ~4 voxels.
2. **Signal Dilution**: When using Global Average Pooling (GAP), tumor features are averaged with background (liver, bowel, air, bone), diluting the diagnostic signal by >90%.
3. **Irrelevant Context**: The model sees mostly non-tumor anatomy.

### Solution: Tumor-Centered Volumes
The `prepare_for_sam3d_roi_cropped()` function:
- **Locates** the tumor using the segmentation mask
- **Crops** a region around it (with margin for context)
- **Resizes/Pads** to standardized 128³ volume
- **Centers** the tumor in every case

## Key Benefits

| Aspect | Full Volume | ROI-Cropped |
|--------|-------------|-------------|
| Small tumor (2cm) | ~4 voxels | ~40 voxels |
| Large tumor (15cm) | ~80 voxels | ~100 voxels (scaled) |
| Background ratio | >95% | <30% |
| Resolution preserved | ❌ | ✅ (for small lesions) |
| Tumor centered | ❌ | ✅ |

## How It Works

### Algorithm
```python
1. Find bounding box of lesion from mask
2. Add margin (default: 10 voxels) for context
3. Crop image and mask to this region
4. If ROI > 128: Resize down proportionally
5. If ROI < 128: Keep original resolution
6. Pad to 128³ (centered)
```

### Example Cases

**Case A: Small GIST (20×15×12 voxels)**
- Original crop with margin: 40×35×32
- After padding: 128×128×128 (lesion at center, original resolution preserved)

**Case B: Large GIST (200×180×150 voxels)**
- Original crop with margin: 220×200×170
- After resize: ~128×115×99
- After padding: 128×128×128 (lesion scaled to fit, still dominates frame)

## Usage

### Basic Usage
```python
from med3pipe.data import prepare_for_sam3d_roi_cropped

prepared, paths = prepare_for_sam3d_roi_cropped(
    dataset_root="path/to/GIST",
    sam3d_root="path/to/SAM-Med3D-main/SAM-Med3D-main",
    category="gist",
    ct_name="ct_GIST_roi",  # Note: different name to avoid overwriting
    target_size=128,
    margin=10,  # voxels around bounding box
)
```

### Custom Parameters
```python
# Larger context (more background)
prepare_for_sam3d_roi_cropped(
    ...,
    margin=20,  # More context around tumor
)

# Different target size
prepare_for_sam3d_roi_cropped(
    ...,
    target_size=96,  # Smaller volumes (faster processing)
)
```

### Integration with Full Pipeline
```python
from med3pipe.pipelines import run_single_dataset

# Use ROI-cropped preparation
result = run_single_dataset(
    method="tabpfn",
    dataset_root="path/to/GIST",
    category="gist",
    ct_name="ct_GIST_roi",  # Match the ROI-cropped data
    # ... other parameters
)
```

## When to Use Which Approach

### Use Full-Volume (`prepare_for_sam3d`)
- Exploring different pooling strategies (GAP vs masked)
- Working with diffuse/multi-focal disease
- Need exact consistency with original SAM-Med3D training

### Use ROI-Cropped (`prepare_for_sam3d_roi_cropped`)
- Small, well-defined lesions (GIST, small metastases)
- Want to preserve lesion resolution
- Using Global Average Pooling (GAP) for features
- **Recommended for tumor classification tasks**

## Clinical Workflow Considerations

### Training
Uses ground-truth segmentation masks to locate tumors → crops around them.

### Inference (Real-World Usage)
In clinical practice with SAM-Med3D:
1. User provides a **click** or **bounding box** prompt (not a dense segmentation)
2. Use that prompt to define the crop region
3. Apply same preprocessing (crop + resize/pad)
4. Extract features and classify

This is **fair** because:
- SAM is designed for interactive prompting
- A radiologist knows where the lesion is (can click on it)
- We don't need pixel-perfect segmentation, just the location

## Technical Details

### Coordinate Systems
- SimpleITK uses **(X, Y, Z)** ordering
- NumPy arrays use **(Z, Y, X)** ordering (after `GetArrayFromImage`)
- The code handles this conversion automatically

### Edge Cases
- **Empty masks**: Skipped with warning
- **Multi-lesion cases**: Uses bounding box of merged mask (covers all lesions)
- **Very large tumors**: Scaled down to fit 128³

### Output Structure
Same as standard preparation:
```
SAM-Med3D-main/SAM-Med3D-main/data/train/gist/ct_GIST_roi/
├── imagesTr/
│   ├── GIST-001_CT.nii.gz  (128×128×128, tumor-centered)
│   └── ...
└── labelsTr/
    ├── GIST-001_CT.nii.gz  (128×128×128, binary mask)
    └── ...
```

## Compatibility

### Works With
- ✅ All existing TabPFN/LoCalPFN pipelines
- ✅ SAM-Med3D feature extraction
- ✅ Classification head training
- ✅ Lesion size filtering

### Requires
- Ground-truth segmentation masks (for training)
- SimpleITK, NumPy

## Performance Impact

**Memory**: Slightly lower (smaller crops in memory during processing)  
**Disk**: Same (outputs are 128³ regardless)  
**Speed**: Slightly faster (less data to resize)  
**Model Accuracy**: Expected to improve (better signal-to-noise ratio)

## Next Steps

After preparing with ROI-cropping:
1. Run feature extraction as usual
2. Use Global Average Pooling (GAP) - now justified because volumes are tumor-centered
3. Train TabPFN classifier
4. Compare results with full-volume approach

## References
- See `med3pipe/data/prepare.py` for implementation
- See `notebooks/` for usage examples (coming soon)
