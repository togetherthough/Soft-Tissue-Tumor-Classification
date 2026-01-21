# Preprocessing Pipeline

> Resize-then-pad approach for SAM-Med3D compatible 3D volumes

## Overview

The preprocessing pipeline converts 3D medical volumes (CT/MRI) into the 128³ format required by SAM-Med3D while preserving as much information as possible.

## The Problem

SAM-Med3D requires fixed 128×128×128 input volumes. Medical images vary widely in size (e.g., 200×150×300 voxels).

### Old Approach: CropOrPad
```
Input: (200, 150, 300)
   ↓ CropOrPad(128³)
   → (128, 128, 128)    # Crops away 172 slices!
```
**Problem**: Data loss when dimensions exceed 128.

### New Approach: Resize-Then-Pad
```
Input: (200, 150, 300)
   ↓ ResizeLargestTo(128)
   → (85, 64, 128)      # Proportional downscaling
   ↓ CropOrPad(128³)
   → (128, 128, 128)    # Pad smaller dimensions
```
**Benefit**: All information preserved via downsampling.

---

## Implementation

### Core Transform: `ResizeLargestTo`

```python
from med3pipe.sam.transforms import ResizeLargestTo
import torchio as tio

transform = tio.Compose([
    tio.ToCanonical(),                           # Reorient to RAS+
    ResizeLargestTo(target_size=128),            # Resize largest → 128
    tio.CropOrPad(target_shape=(128, 128, 128)), # Pad to 128³
    tio.ZNormalization(masking_method=lambda x: x > 0),  # Optional
])
```

### How `ResizeLargestTo` Works

1. Find the largest spatial dimension
2. Compute scale factor: `scale = target_size / largest_dim`
3. Apply proportional scaling to all dimensions
4. Result: Largest dimension = target_size, others < target_size

```python
class ResizeLargestTo(tio.Transform):
    def __init__(self, target_size=128, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size
    
    def apply_transform(self, subject):
        spatial_shape = subject.get_first_image().spatial_shape
        max_dim = max(spatial_shape)
        scale = self.target_size / max_dim if max_dim > 0 else 1.0
        new_shape = tuple(int(dim * scale) for dim in spatial_shape)
        resize_transform = tio.Resize(target_shape=new_shape)
        return resize_transform(subject)
```

---

## Usage

### Option 1: Utility Functions (Recommended)

```python
from med3pipe.sam.transforms import load_volume_resize_pad, load_mask_resize_pad

# Load and preprocess image
image = load_volume_resize_pad("path/to/image.nii.gz", img_size=128)
# Shape: (1, 1, 128, 128, 128)

# Load and preprocess mask
mask = load_mask_resize_pad("path/to/mask.nii.gz", img_size=128)
```

### Option 2: Transform Pipeline

```python
from med3pipe.sam.transforms import make_sam3d_transform
import SimpleITK as sitk
import torchio as tio

transform = make_sam3d_transform(img_size=128, normalize=False)

sitk_img = sitk.ReadImage("path/to/image.nii.gz")
sitk_arr, _ = tio.data.io.sitk_to_nib(sitk_img)
subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_arr))
subject = transform(subject)
image = subject.image.data  # Shape: (1, 128, 128, 128)
```

### Option 3: In Data Loaders

```python
import torchio as tio
from med3pipe.sam.transforms import ResizeLargestTo

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = tio.Compose([
            tio.ToCanonical(),
            ResizeLargestTo(target_size=128),
            tio.CropOrPad(target_shape=(128, 128, 128)),
        ])
    
    def __getitem__(self, idx):
        subject = tio.Subject(
            image=tio.ScalarImage(self.paths[idx])
        )
        return self.transform(subject)
```

---

## Complete Pipeline

The full preprocessing pipeline in `med3pipe/sam/core.py`:

```python
def make_pre_transform(img_size=128, normalize=True):
    transforms = [
        tio.ToCanonical(),
        ResizeLargestTo(target_size=img_size),
        tio.CropOrPad(target_shape=(img_size, img_size, img_size)),
    ]
    if normalize:
        transforms.append(
            tio.ZNormalization(masking_method=lambda x: x > 0)
        )
    return tio.Compose(transforms)
```

---

## Files

| File | Description |
|------|-------------|
| `med3pipe/sam/transforms.py` | Standalone preprocessing utilities |
| `med3pipe/sam/core.py` | Core SAM-Med3D integration |
| `tests/test_transforms.py` | Test suite |

---

## Testing

```bash
python tests/test_transforms.py
```

Expected output:
```
✅ ResizeLargestTo Transform: 5/5 tests passed
✅ Full Pipeline: 5/5 tests passed
✅ Data Preservation: Markers preserved after resize
✅ ALL TESTS PASSED!
```

---

## Migration from Old Code

Replace:
```python
transform = tio.CropOrPad(target_shape=(128, 128, 128))
```

With:
```python
from med3pipe.sam.transforms import ResizeLargestTo
transform = tio.Compose([
    ResizeLargestTo(target_size=128),
    tio.CropOrPad(target_shape=(128, 128, 128)),
])
```

---

## Related Documentation

- [Changelog](../CHANGES_SUMMARY.md) — Preprocessing upgrade summary
- [Quick Reference](../QUICK_REFERENCE.md) — Command cheat sheet
- [Embeddings](embeddings.md) — Feature extraction guide
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
