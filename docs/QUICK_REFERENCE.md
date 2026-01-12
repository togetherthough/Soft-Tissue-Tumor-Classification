# Quick Reference: Resize-Then-Pad Preprocessing

## TL;DR

**Before**: `tio.CropOrPad((128,128,128))` → loses data if volume > 128  
**After**: `ResizeLargestTo(128) → CropOrPad((128,128,128))` → preserves all data

## Import & Use

```python
from med3pipe.sam.transforms import load_volume_resize_pad, load_mask_resize_pad

# Load image with new preprocessing (one-liner)
image = load_volume_resize_pad("path/to/image.nii.gz", img_size=128)

# Load mask with new preprocessing
mask = load_mask_resize_pad("path/to/mask.nii.gz", img_size=128)
```

## Manual Transform

```python
import torchio as tio
from med3pipe.sam.transforms import ResizeLargestTo

transform = tio.Compose([
    tio.ToCanonical(),
    ResizeLargestTo(target_size=128),              # NEW: Resize (largest → 128)
    tio.CropOrPad(target_shape=(128, 128, 128)),  # Pad to 128³
])

subject = tio.Subject(image=tio.ScalarImage.from_sitk(sitk_image))
result = transform(subject)
```

## In Data Loaders

```python
from med3pipe.sam.transforms import ResizeLargestTo

# OLD
transform = tio.Compose([
    tio.ToCanonical(),
    tio.CropOrPad(target_shape=(128, 128, 128)),
])

# NEW
transform = tio.Compose([
    tio.ToCanonical(),
    ResizeLargestTo(target_size=128),              # Add this line
    tio.CropOrPad(target_shape=(128, 128, 128)),
])
```

## How It Works

```
Input: (200, 150, 100)
   ↓ ResizeLargestTo(128)
   → (128, 96, 64)      # Largest dim (200) → 128, others scaled proportionally
   ↓ CropOrPad(128³)
   → (128, 128, 128)    # Pad smaller dims to 128
```

## Key Files

- **`med3pipe/sam/transforms.py`** - Standalone utilities (recommended)
- **`med3pipe/sam/core.py`** - Core SAM pipeline (auto-uses new approach)
- **`docs/PREPROCESSING_UPGRADE.md`** - Full documentation
- **`tests/test_transforms.py`** - Run tests: `python tests/test_transforms.py`

## Benefits

✅ **Zero data loss** - all info preserved via downsampling  
✅ **Aspect ratio preserved** - proportional scaling  
✅ **SAM-Med3D compatible** - still outputs 128³  
✅ **Better for anisotropic volumes** - handles different sizes gracefully

## Test It

```bash
python tests/test_transforms.py
```

Should show:
```
✅ ALL TESTS PASSED!
The new resize-then-pad approach is working correctly.
```
