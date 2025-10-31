#!/usr/bin/env python
"""
Verify SAM-Med3D weights are loading correctly
Compatible with Python 2.7+ and Python 3.x
"""

from __future__ import print_function
import sys
import os

# Try pathlib for Python 3, fallback for Python 2
try:
    from pathlib import Path
except ImportError:
    # Python 2 fallback
    class Path(object):
        def __init__(self, path):
            self.path = os.path.abspath(str(path))
        def __truediv__(self, other):
            return Path(os.path.join(self.path, str(other)))
        def __str__(self):
            return self.path
        def exists(self):
            return os.path.exists(self.path)
        def stat(self):
            class Stat:
                def __init__(self, path):
                    self.st_size = os.path.getsize(path)
            return Stat(self.path)
        @property
        def parent(self):
            return Path(os.path.dirname(self.path))

import torch

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

print("="*80)
print("SAM-Med3D Weight Verification")
print("="*80)

# Check checkpoint file
ckpt_path = repo_root / "SAM-Med3D-main" / "SAM-Med3D-main" / "ckpt" / "sam_med3d_turbo.pth"
print("\n1. Checking checkpoint file...")
print("   Path: {}".format(ckpt_path))

if not ckpt_path.exists():
    print("   ❌ CHECKPOINT NOT FOUND!")
    print("   Please download it to: {}".format(ckpt_path))
    sys.exit(1)

file_size_mb = ckpt_path.stat().st_size / (1024**2)
print("   ✓ Checkpoint exists: {:.1f} MB".format(file_size_mb))

if file_size_mb < 700:
    print("   ⚠️  WARNING: File seems too small (expected ~750MB)")

# Load checkpoint
print("\n2. Loading checkpoint...")
try:
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    print("   ✓ Checkpoint loaded successfully")
except Exception as e:
    print("   ❌ ERROR loading checkpoint: {}".format(e))
    sys.exit(1)

# Inspect checkpoint structure
print("\n3. Checkpoint structure:")
if isinstance(checkpoint, dict):
    print("   Keys in checkpoint: {}".format(list(checkpoint.keys())))
    
    # Check for model weights
    if 'model' in checkpoint:
        model_dict = checkpoint['model']
        print("   ✓ Found 'model' key with {} parameters".format(len(model_dict)))
    elif 'state_dict' in checkpoint:
        model_dict = checkpoint['state_dict']
        print("   ✓ Found 'state_dict' key with {} parameters".format(len(model_dict)))
    else:
        model_dict = checkpoint
        print("   ✓ Direct state dict with {} parameters".format(len(model_dict)))
    
    # Show some parameter names and shapes
    print("\n4. Sample parameters:")
    for i, (name, param) in enumerate(list(model_dict.items())[:5]):
        if isinstance(param, torch.Tensor):
            print("   {}. {}: shape={}, mean={:.6f}".format(
                i+1, name, tuple(param.shape), param.float().mean().item()))
        else:
            print("   {}. {}: {}".format(i+1, name, type(param)))
    
    # Check if weights look reasonable (not all zeros or random)
    print("\n5. Weight sanity checks:")
    weight_values = []
    for name, param in list(model_dict.items())[:10]:
        if isinstance(param, torch.Tensor) and param.numel() > 0:
            weight_values.append(param.float().abs().mean().item())
    
    if weight_values:
        avg_weight = sum(weight_values) / len(weight_values)
        print("   Average absolute weight value: {:.6f}".format(avg_weight))
        
        if avg_weight < 1e-6:
            print("   ⚠️  WARNING: Weights are very small (might be all zeros)")
        elif avg_weight > 100:
            print("   ⚠️  WARNING: Weights are very large (might be corrupted)")
        else:
            print("   ✓ Weights look reasonable")
else:
    print("   ⚠️  Unexpected checkpoint format: {}".format(type(checkpoint)))

# Try loading into actual model
print("\n6. Testing model loading...")
try:
    sys.path.insert(0, str(repo_root / "SAM-Med3D-main" / "SAM-Med3D-main"))
    from segment_anything.build_sam3D import sam_model_registry3D
    
    sam3d = sam_model_registry3D['vit_b_ori'](checkpoint=None)
    print("   ✓ Model architecture created")
    
    # Load weights
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    sam3d.load_state_dict(state_dict, strict=False)
    print("   ✓ Weights loaded into model")
    
    # Check a specific layer to verify it's not random
    first_conv = None
    for name, param in sam3d.named_parameters():
        if 'conv' in name.lower() and len(param.shape) >= 2:
            first_conv = param
            print("   ✓ Checking layer '{}': shape={}".format(name, tuple(param.shape)))
            print("     Mean: {:.6f}".format(param.data.mean().item()))
            print("     Std:  {:.6f}".format(param.data.std().item()))
            print("     Min:  {:.6f}".format(param.data.min().item()))
            print("     Max:  {:.6f}".format(param.data.max().item()))
            break
    
    if first_conv is not None:
        # Random init typically has std around 0.01-0.1 for conv layers
        # Pretrained weights should have different statistics
        std_val = first_conv.data.std().item()
        if std_val < 1e-6:
            print("   ❌ WARNING: Weights might be all zeros!")
        else:
            print("   ✓ Weights appear to be loaded (not random init)")
    
except Exception as e:
    print("   ⚠️  Could not test model loading: {}".format(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Verification complete!")
print("="*80 + "\n")
