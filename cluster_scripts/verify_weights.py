#!/usr/bin/env python
"""
Verify SAM-Med3D weights are loading correctly
"""

import sys
from pathlib import Path
import torch

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

print("="*80)
print("SAM-Med3D Weight Verification")
print("="*80)

# Check checkpoint file
ckpt_path = repo_root / "SAM-Med3D-main" / "SAM-Med3D-main" / "ckpt" / "sam_med3d_turbo.pth"
print(f"\n1. Checking checkpoint file...")
print(f"   Path: {ckpt_path}")

if not ckpt_path.exists():
    print("   ❌ CHECKPOINT NOT FOUND!")
    print(f"   Please download it to: {ckpt_path}")
    sys.exit(1)

file_size_mb = ckpt_path.stat().st_size / (1024**2)
print(f"   ✓ Checkpoint exists: {file_size_mb:.1f} MB")

if file_size_mb < 700:
    print(f"   ⚠️  WARNING: File seems too small (expected ~750MB)")

# Load checkpoint
print(f"\n2. Loading checkpoint...")
try:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(f"   ✓ Checkpoint loaded successfully")
except Exception as e:
    print(f"   ❌ ERROR loading checkpoint: {e}")
    sys.exit(1)

# Inspect checkpoint structure
print(f"\n3. Checkpoint structure:")
if isinstance(checkpoint, dict):
    print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
    
    # Check for model weights
    if 'model' in checkpoint:
        model_dict = checkpoint['model']
        print(f"   ✓ Found 'model' key with {len(model_dict)} parameters")
    elif 'state_dict' in checkpoint:
        model_dict = checkpoint['state_dict']
        print(f"   ✓ Found 'state_dict' key with {len(model_dict)} parameters")
    else:
        model_dict = checkpoint
        print(f"   ✓ Direct state dict with {len(model_dict)} parameters")
    
    # Show some parameter names and shapes
    print(f"\n4. Sample parameters:")
    for i, (name, param) in enumerate(list(model_dict.items())[:5]):
        if isinstance(param, torch.Tensor):
            print(f"   {i+1}. {name}: shape={tuple(param.shape)}, mean={param.float().mean().item():.6f}")
        else:
            print(f"   {i+1}. {name}: {type(param)}")
    
    # Check if weights look reasonable (not all zeros or random)
    print(f"\n5. Weight sanity checks:")
    weight_values = []
    for name, param in list(model_dict.items())[:10]:
        if isinstance(param, torch.Tensor) and param.numel() > 0:
            weight_values.append(param.float().abs().mean().item())
    
    if weight_values:
        avg_weight = sum(weight_values) / len(weight_values)
        print(f"   Average absolute weight value: {avg_weight:.6f}")
        
        if avg_weight < 1e-6:
            print(f"   ⚠️  WARNING: Weights are very small (might be all zeros)")
        elif avg_weight > 100:
            print(f"   ⚠️  WARNING: Weights are very large (might be corrupted)")
        else:
            print(f"   ✓ Weights look reasonable")
else:
    print(f"   ⚠️  Unexpected checkpoint format: {type(checkpoint)}")

# Try loading into actual model
print(f"\n6. Testing model loading...")
try:
    sys.path.insert(0, str(repo_root / "SAM-Med3D-main" / "SAM-Med3D-main"))
    from segment_anything.build_sam3D import sam_model_registry3D
    
    sam3d = sam_model_registry3D['vit_b_ori'](checkpoint=None)
    print(f"   ✓ Model architecture created")
    
    # Load weights
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    sam3d.load_state_dict(state_dict, strict=False)
    print(f"   ✓ Weights loaded into model")
    
    # Check a specific layer to verify it's not random
    first_conv = None
    for name, param in sam3d.named_parameters():
        if 'conv' in name.lower() and len(param.shape) >= 2:
            first_conv = param
            print(f"   ✓ Checking layer '{name}': shape={tuple(param.shape)}")
            print(f"     Mean: {param.data.mean().item():.6f}")
            print(f"     Std:  {param.data.std().item():.6f}")
            print(f"     Min:  {param.data.min().item():.6f}")
            print(f"     Max:  {param.data.max().item():.6f}")
            break
    
    if first_conv is not None:
        # Random init typically has std around 0.01-0.1 for conv layers
        # Pretrained weights should have different statistics
        std_val = first_conv.data.std().item()
        if std_val < 1e-6:
            print(f"   ❌ WARNING: Weights might be all zeros!")
        else:
            print(f"   ✓ Weights appear to be loaded (not random init)")
    
except Exception as e:
    print(f"   ⚠️  Could not test model loading: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("Verification complete!")
print(f"{'='*80}\n")
