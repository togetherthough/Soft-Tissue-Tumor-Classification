# SAM-Med3D Checkpoint Loading Guide

## ✅ Updated Checkpoint Loading

The script now automatically searches for checkpoints in multiple locations and provides helpful feedback.

## 📁 Automatic Search Locations

The script checks these paths in order:

1. `SAM-Med3D-main/SAM-Med3D-main/work_dir/SAM/sam_model_best.pth`
2. `SAM-Med3D-main/SAM-Med3D-main/work_dir/sam_model_best.pth`
3. `SAM-Med3D-main/SAM-Med3D-main/checkpoints/sam_model_best.pth`
4. `SAM-Med3D-main/SAM-Med3D-main/sam_model_best.pth`
5. `SAM-Med3D-main/SAM-Med3D-main/work_dir/SAM-Med3D/sam_model_best.pth`

## 🎯 Manual Checkpoint Specification

If your checkpoint is in a custom location, set the `MANUAL_CHECKPOINT` variable:

```python
# At the top of the script (around line 45)
MANUAL_CHECKPOINT = Path("C:/path/to/your/checkpoint.pth")
```

Or in a notebook cell:
```python
# Option 1: Absolute path
MANUAL_CHECKPOINT = Path("C:/Users/yourname/Desktop/sam_model_best.pth")

# Option 2: Relative to project
MANUAL_CHECKPOINT = project_root / "models" / "sam_model_best.pth"

# Option 3: Disable (use auto-search)
MANUAL_CHECKPOINT = None
```

## 📥 Downloading Checkpoints

If you don't have a checkpoint, download the official SAM-Med3D checkpoint:

```bash
# From the SAM-Med3D repository
cd SAM-Med3D-main/SAM-Med3D-main
mkdir -p work_dir/SAM

# Download the checkpoint (example - use actual download link)
wget https://path-to-checkpoint/sam_model_best.pth -P work_dir/SAM/
```

Or manually:
1. Download from: [SAM-Med3D releases](https://github.com/uni-medical/SAM-Med3D)
2. Place in: `SAM-Med3D-main/SAM-Med3D-main/work_dir/SAM/sam_model_best.pth`

## 🔍 Console Output

### ✅ Checkpoint Found
```
✅ Found checkpoint: .../work_dir/SAM/sam_model_best.pth
📦 Loading checkpoint from: .../work_dir/SAM/sam_model_best.pth
✅ SAM-Med3D model loaded successfully!
```

### ⚠️ Checkpoint Not Found
```
⚠️  No checkpoint found. Loading model without pretrained weights.
   Searched locations:
   - .../work_dir/SAM/sam_model_best.pth
   - .../work_dir/sam_model_best.pth
   ...
   
   💡 Tip: Set MANUAL_CHECKPOINT variable to specify checkpoint manually
✅ SAM-Med3D model loaded successfully!
```

**Note**: Model will still load, but without pretrained weights (poor segmentation performance).

### 📦 Manual Checkpoint
```
✅ Using manually specified checkpoint: C:/custom/path/checkpoint.pth
📦 Loading checkpoint from: C:/custom/path/checkpoint.pth
✅ SAM-Med3D model loaded successfully!
```

## 🔧 Troubleshooting

### Problem: Checkpoint Not Found

**Solution 1**: Check if file exists
```python
from pathlib import Path
checkpoint = Path("SAM-Med3D-main/SAM-Med3D-main/work_dir/SAM/sam_model_best.pth")
print(f"Exists: {checkpoint.exists()}")
print(f"Absolute path: {checkpoint.absolute()}")
```

**Solution 2**: List available files
```python
from pathlib import Path
sam3d_root = Path("SAM-Med3D-main/SAM-Med3D-main")
print("Files in work_dir:")
for f in (sam3d_root / "work_dir").rglob("*.pth"):
    print(f"  {f}")
```

**Solution 3**: Use manual specification
```python
MANUAL_CHECKPOINT = Path("C:/full/absolute/path/to/checkpoint.pth")
```

### Problem: Wrong Checkpoint Loaded

Check which checkpoint was loaded:
```python
# The script prints this automatically:
# 📦 Loading checkpoint from: ...
```

Verify it's the correct one, then proceed.

### Problem: Checkpoint Loads but Segmentation is Bad

This might mean:
1. ❌ Wrong model type (`vit_b_ori` vs `vit_b` vs `vit_l`)
2. ❌ Checkpoint is for different task
3. ❌ Checkpoint is corrupted

Try different model types:
```python
# In the build_sam3d_model call, change model_type:
model = build_sam3d_model(
    sam3d_root=sam3d_root,
    model_type="vit_b",  # Try: "vit_b", "vit_b_ori", "vit_l", "vit_h"
    checkpoint=checkpoint_path,
    device=device,
    eval_mode=True
)
```

## 📝 Summary

✅ **Auto-detection**: Works for standard SAM-Med3D directory structures  
✅ **Manual override**: Use `MANUAL_CHECKPOINT` for custom locations  
✅ **Helpful feedback**: Console shows exactly what's happening  
✅ **Graceful fallback**: Loads model even without checkpoint (but won't segment well)  

---

**Location in code**: Lines 44-99 in `SAM_Visualization_Fixed.py`
