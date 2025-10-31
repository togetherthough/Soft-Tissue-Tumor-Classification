# Adding Multi-View Visualization to SAM_Visualization_Fixed.ipynb

## Quick Start - Two Options:

### **Option 1: Import the plotting functions** (Recommended)

Add this cell after the imports section in `SAM_Visualization_Fixed.ipynb`:

```python
# Import multi-view plotting functions
from multiview_plotting import plot_all_anatomical_views, plot_single_view
```

Then replace the plotting call in the main processing loop:

**BEFORE** (line ~1037):
```python
fig = plot_coronal_slices(
    image_vol_display,
    seg_mask, 
    gt_mask,
    title=f"{dataset_name.upper()}: {img_path.stem}",
    n_slices=5,
    modality=modality
)
```

**AFTER - Option A (All 3 views at once)**:
```python
# Show all anatomical views (Axial, Sagittal, Coronal)
fig = plot_all_anatomical_views(
    image_vol_display,
    seg_mask,
    gt_mask,
    title=f"{dataset_name.upper()}: {img_path.stem}",
    n_slices=3,  # 3 slices per view = 9 total slices
    modality=modality
)
```

**AFTER - Option B (One view at a time)**:
```python
# Show each view separately
for view_type in ['axial', 'sagittal', 'coronal']:
    fig = plot_single_view(
        image_vol_display,
        seg_mask,
        gt_mask,
        view_type=view_type,
        title=f"{dataset_name.upper()}: {img_path.stem}",
        n_slices=5,
        modality=modality
    )
    plt.show()
```

---

### **Option 2: Copy the functions directly into the notebook**

1. Open `multiview_plotting.py`
2. Copy the two functions: `plot_all_anatomical_views()` and `plot_single_view()`
3. Paste them into a new code cell in `SAM_Visualization_Fixed.ipynb` (after the existing plotting function)
4. Then use the same calls as above

---

## Anatomical View Explanations:

| View | Dimension | What You See | Slicing Axis |
|------|-----------|--------------|--------------|
| **AXIAL** | 0 (D/Z) | Top → Bottom view (looking down through body/head) | Z-axis |
| **SAGITTAL** | 2 (W/X) | Left → Right view (side view of body) | X-axis |
| **CORONAL** | 1 (H/Y) | Front → Back view (face-on view) | Y-axis |

## Example Output:

When you use `plot_all_anatomical_views()`, you'll get a figure with:
- **3 rows** (one per anatomical view)
- **n_slices columns** (default 3)
- **3 panels per slice**: Original | SAM Segmentation | Ground Truth

Total subplots = 3 views × 3 slices × 3 panels = **27 images** showing comprehensive visualization!

## Customization:

### Adjust number of slices per view:
```python
plot_all_anatomical_views(..., n_slices=5)  # 5 slices per view = 15 total slices
```

### Show only specific views:
```python
# Just axial view
plot_single_view(..., view_type='axial', n_slices=7)

# Just coronal view (original behavior)
plot_single_view(..., view_type='coronal', n_slices=5)
```

### Remove ground truth overlay:
```python
plot_all_anatomical_views(
    image_vol_display,
    seg_mask,
    gt_mask=None,  # No ground truth
    ...
)
```

---

## Benefits:

✅ **Complete anatomical coverage** - see the full 3D structure  
✅ **Better understanding** of SAM segmentation quality from all angles  
✅ **Identify issues** that may only be visible in certain views  
✅ **Standard medical imaging practice** - radiologists view all 3 planes  

---

## File Locations:

- **Plotting functions**: `notebooks/multiview_plotting.py`
- **This guide**: `notebooks/MULTIVIEW_GUIDE.md`
- **Original notebook**: `notebooks/SAM_Visualization_Fixed.ipynb`

---

## Quick Test:

Add a new cell at the end of the notebook to test:

```python
# Quick test - visualize one sample with all views
from multiview_plotting import plot_all_anatomical_views

test_img_path = images[0]
test_label_path = labels[0]

img_sam = load_volume_for_sam(test_img_path)
img_display = load_volume_for_display(test_img_path)
gt = load_mask_for_sam(test_label_path)
seg, _ = generate_sam_with_bbox_prompt(model, img_sam, gt, device, use_single_point=True)

fig = plot_all_anatomical_views(
    img_display, seg, gt,
    title="Multi-View Test",
    n_slices=3,
    modality="CT"
)
plt.show()
```
