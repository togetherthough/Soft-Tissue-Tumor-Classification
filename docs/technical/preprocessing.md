# Preprocessing Upgrade: Resize-Then-Pad for SAM-Med3D

## Overview

The preprocessing pipeline has been upgraded to use a **resize-then-pad** approach instead of the previous **crop-or-pad** approach. This minimizes data loss while still achieving the required 128³ format for SAM-Med3D.

## Why the Change?

### Previous Approach (CropOrPad)
- **Problem**: Directly crops or pads to 128³
- **Data Loss**: If the original volume is larger than 128 in any dimension, that data is **cropped away**
- **Example**: A volume of size (200, 150, 100) would lose 72 slices in the first dimension

### New Approach (Resize-Then-Pad)
- **Step 1**: Resize so the **largest dimension becomes 128** (preserves aspect ratio)
- **Step 2**: Pad the remaining smaller dimensions to reach 128³
- **Benefit**: Minimal data loss - all original data is preserved, just scaled down
- **Example**: A volume of size (200, 150, 100) becomes (128, 96, 64) after resize, then (128, 128, 128) after padding

## Implementation

### Core Transform: `ResizeLargestTo`

A new TorchIO transform that resizes volumes to make the largest dimension equal to a target size:

```python
from med3pipe.sam.transforms import ResizeLargestTo

# Resize so largest dimension = 128
transform = ResizeLargestTo(target_size=128)
```

### Complete Pipeline

```python
import torchio as tio
from med3pipe.sam.transforms import ResizeLargestTo

transform = tio.Compose([
    tio.ToCanonical(),                                    # Reorient to RAS+
    ResizeLargestTo(target_size=128),                    # Resize (largest dim → 128)
    tio.CropOrPad(target_shape=(128, 128, 128)),         # Pad to 128³
    tio.ZNormalization(masking_method=lambda x: x > 0),  # Normalize (optional)
])
```

## Usage

### Option 1: Use the new utility functions

```python
from med3pipe.sam.transforms import load_volume_resize_pad, load_mask_resize_pad

# Load image with resize-then-pad preprocessing
image = load_volume_resize_pad(img_path, img_size=128, normalize=False)

# Load mask with resize-then-pad preprocessing  
mask = load_mask_resize_pad(mask_path, img_size=128)
```

### Option 2: Use the transform directly

```python
from med3pipe.sam.transforms import make_sam3d_transform
import torchio as tio
import SimpleITK as sitk

# Create transform
transform = make_sam3d_transform(img_size=128, normalize=False)

# Load and transform
sitk_img = sitk.ReadImage(str(img_path))
sitk_arr, _ = tio.data.io.sitk_to_nib(sitk_img)
subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_arr))
subject = transform(subject)
image = subject.image.data
```

### Option 3: Use in data loader

```python
import torchio as tio
from med3pipe.sam.transforms import ResizeLargestTo

# In your dataset or data loader
transform = tio.Compose([
    tio.ToCanonical(),
    ResizeLargestTo(target_size=128),
    tio.CropOrPad(target_shape=(128, 128, 128)),
])
```

## Files Updated

1. **`med3pipe/sam/core.py`**
   - Added `ResizeLargestTo` transform class
   - Updated `make_pre_transform()` to use resize-then-pad approach

2. **`SAM-Med3D-main/SAM-Med3D-main/utils/data_loader.py`**
   - Added `ResizeLargestTo` transform class
   - Updated example usage in `__main__` section

3. **`med3pipe/sam/transforms.py`** (NEW)
   - Standalone module with all preprocessing utilities
   - Includes `ResizeLargestTo`, `make_sam3d_transform`, `load_volume_resize_pad`, `load_mask_resize_pad`

## Updating Notebooks

For Jupyter notebooks using the old approach, replace:

```python
# OLD: Direct CropOrPad
def load_volume_croponly(img_path, img_size=128):
    sitk_img = sitk.ReadImage(str(img_path))
    sitk_arr, _ = tio.data.io.sitk_to_nib(sitk_img)
    subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_arr))
    crop_transform = tio.CropOrPad(target_shape=(img_size, img_size, img_size))
    subject = crop_transform(subject)
    image = subject.image.data.clone().detach()
    image = image.unsqueeze(0)
    image = image.float()
    return image
```

With:

```python
# NEW: Resize-then-pad
from med3pipe.sam.transforms import load_volume_resize_pad

# Simple one-liner
image = load_volume_resize_pad(img_path, img_size=128, normalize=False)
```

Or if you prefer to keep the function definition:

```python
# NEW: Resize-then-pad (manual)
class ResizeLargestTo(tio.Transform):
    def __init__(self, target_size=128, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size
    
    def apply_transform(self, subject):
        first_image = subject.get_first_image()
        spatial_shape = first_image.spatial_shape
        max_dim = max(spatial_shape)
        scale = self.target_size / max_dim if max_dim > 0 else 1.0
        new_shape = tuple(int(dim * scale) for dim in spatial_shape)
        resize_transform = tio.Resize(target_shape=new_shape)
        return resize_transform(subject)

def load_volume_resize_pad(img_path, img_size=128):
    sitk_img = sitk.ReadImage(str(img_path))
    sitk_arr, _ = tio.data.io.sitk_to_nib(sitk_img)
    subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_arr))
    transform = tio.Compose([
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
    ])
    subject = transform(subject)
    image = subject.image.data.clone().detach()
    image = image.unsqueeze(0)
    image = image.float()
    return image
```

## Benefits Summary

✅ **Preserves all data**: No cropping, just scaling  
✅ **Minimal padding**: Only pads dimensions smaller than 128  
✅ **Maintains aspect ratio**: Proportional resizing  
✅ **SAM-Med3D compatible**: Still produces 128³ volumes  
✅ **Better for anisotropic volumes**: Handles different dimension sizes gracefully  

## Example Comparison

| Original Size | Old (CropOrPad) | New (Resize→Pad) | Data Preserved |
|---------------|-----------------|------------------|----------------|
| (200, 150, 100) | (128, 128, 128) | (128, 128, 128) | 100% vs ~40% |
| (80, 60, 50) | (128, 128, 128) | (128, 128, 128) | 100% (both pad) |
| (128, 200, 100) | (128, 128, 128) | (128, 128, 128) | 100% vs ~64% |

In the new approach:
- (200, 150, 100) → resize → (128, 96, 64) → pad → (128, 128, 128)
- All original information preserved, just downsampled to fit
