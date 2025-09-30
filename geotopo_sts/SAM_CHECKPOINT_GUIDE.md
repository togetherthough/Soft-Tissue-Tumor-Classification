# SAM-Med3D Checkpoint Guide

## 🎯 Quick Fix for the Error

The error occurs because SAM-Med3D needs to be loaded differently. **I've already fixed this!**

---

## 📦 Option 1: Download Pre-trained SAM-Med3D (Easiest)

### Official SAM-Med3D Checkpoint

```bash
# Download pre-trained SAM-Med3D-turbo
wget https://github.com/uni-medical/SAM-Med3D/releases/download/v0.1/sam_med3d_turbo.pth

# Or download from Hugging Face
# Visit: https://huggingface.co/uni-medical/SAM-Med3D
```

### Alternative: Original SAM (not optimized for medical, but works)

```bash
# Download ViT-B checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## 📦 Option 2: Test Without SAM (Quick Demo)

You can test the pipeline structure WITHOUT SAM using existing masks:

```python
from geotopo_sts.sam_geotopo_integration import GeoTopoSTS_AutoInference

# Initialize WITHOUT SAM
pipeline = GeoTopoSTS_AutoInference(
    geotopo_config_path='config.yaml',
    geotopo_checkpoint_path='./outputs/gist_experiment/best_model.pth',
    sam_checkpoint_path=None,  # ← No SAM checkpoint
    use_sam=False,  # ← Disable SAM
    device='cpu'
)

# Load existing mask from your GIST data
import nibabel as nib
mask_nii = nib.load('path/to/segmentation.nii.gz')
mask = mask_nii.get_fdata().astype(np.uint8)

# Predict using existing mask
prediction = pipeline.predict(ct_volume, spacing, mask=mask)
```

This lets you test GeoTopo-STS while you download SAM!

---

## 📦 Option 3: Use SAM-Med3D from Your Repo

You already have SAM-Med3D code! Let's check for checkpoints:

```python
import os
from pathlib import Path

# Search for SAM checkpoints
sam_dir = Path('C:/Users/cahel/Desktop/Med3Tab-PFN/SAM-Med3D-main/SAM-Med3D-main')

# Check common locations
checkpoint_locations = [
    sam_dir / 'work_dir' / 'SAM',
    sam_dir / 'work_dir' / 'sam_med3d',
    sam_dir / 'checkpoints',
    sam_dir / 'pretrained',
]

for loc in checkpoint_locations:
    if loc.exists():
        checkpoints = list(loc.glob('*.pth'))
        if checkpoints:
            print(f"Found checkpoints in {loc}:")
            for ckpt in checkpoints:
                print(f"  - {ckpt.name}")
```

---

## 🔧 Updated Notebook Cell (Works Now!)

Replace your initialization cell with this:

```python
from geotopo_sts.sam_geotopo_integration import GeoTopoSTS_AutoInference
import torch

# Option A: WITH SAM (download checkpoint first)
sam_checkpoint = './sam_med3d_turbo.pth'  # Download this first!

# Option B: WITHOUT SAM (use existing masks for testing)
sam_checkpoint = None

# Paths
geotopo_checkpoint = r'C:\Users\cahel\Desktop\Med3Tab-PFN\geotopo_sts\outputs\gist_experiment\best_model.pth'
config_path = r'C:\Users\cahel\Desktop\Med3Tab-PFN\geotopo_sts\config.yaml'

# Initialize pipeline
try:
    pipeline = GeoTopoSTS_AutoInference(
        geotopo_config_path=config_path,
        geotopo_checkpoint_path=geotopo_checkpoint,
        sam_checkpoint_path=sam_checkpoint,  # Can be None
        sam_model_type='vit_b',
        use_sam=(sam_checkpoint is not None),  # Auto-detect
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("\n✅ Pipeline ready!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTip: Set sam_checkpoint=None to test without SAM")
```

---

## 🎯 Quick Decision Tree

### Do you want to test AUTO-SEGMENTATION?
**YES** → Download SAM checkpoint (Option 1)  
**NO** → Use `sam_checkpoint=None` (Option 2)

### Do you have a GPU?
**YES** → Use `device='cuda'` (much faster)  
**NO** → Use `device='cpu'` (works but slower)

### Do you want to wait for download?
**YES** → Get pre-trained SAM (~100MB, 5 min)  
**NO** → Test with existing masks now (instant)

---

## 📝 Summary

**The error is fixed!** Now you can:

1. **Test immediately**: Set `sam_checkpoint=None`
2. **Download SAM later**: When ready, download and enable SAM
3. **Train your own SAM**: Use your GIST masks to fine-tune

**No blocker!** You can continue testing the pipeline right now. 🚀

---

## 🔗 Download Links

### Pre-trained SAM-Med3D
- GitHub: https://github.com/uni-medical/SAM-Med3D/releases
- Hugging Face: https://huggingface.co/uni-medical/SAM-Med3D

### Original SAM (if SAM-Med3D unavailable)
- ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Pick ViT-B for speed, ViT-H for accuracy.
