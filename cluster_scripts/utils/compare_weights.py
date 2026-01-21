#!/usr/bin/env python
"""
Compare loaded model weights vs random initialization
This proves the checkpoint is actually being used
"""

import sys
from pathlib import Path
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sam-med3d"))

from segment_anything.build_sam3D import sam_model_registry3D

print("="*80)
print("Weight Comparison Test")
print("="*80)

ckpt_path = repo_root / "sam-med3d" / "ckpt" / "sam_med3d_turbo.pth"

print("\n1. Creating model with random initialization...")
model_random = sam_model_registry3D['vit_b_ori'](checkpoint=None)

print("\n2. Creating model with pretrained weights...")
if not ckpt_path.exists():
    print(f"❌ Checkpoint not found at: {ckpt_path}")
    sys.exit(1)

model_pretrained = sam_model_registry3D['vit_b_ori'](checkpoint=str(ckpt_path))

print("\n3. Comparing weights...")
diff_count = 0
same_count = 0

for (name1, param1), (name2, param2) in zip(
    model_random.named_parameters(), 
    model_pretrained.named_parameters()
):
    if name1 == name2:
        # Check if weights are different
        if not torch.allclose(param1, param2, atol=1e-6):
            diff_count += 1
        else:
            same_count += 1

print(f"\n   Parameters with different values: {diff_count}")
print(f"   Parameters with same values: {same_count}")

if diff_count == 0:
    print(f"\n   ❌ WARNING: All weights are the same!")
    print(f"      The checkpoint might not be loading correctly!")
elif diff_count > 0 and same_count == 0:
    print(f"\n   ✓ GOOD: Pretrained weights are loaded correctly!")
else:
    print(f"\n   ⚠️  Mixed result. Some weights loaded, some not.")

# Show specific examples
print(f"\n4. Example weight comparison:")
for name, param_rand in list(model_random.named_parameters())[:3]:
    param_pre = dict(model_pretrained.named_parameters())[name]
    
    print(f"\n   Layer: {name}")
    print(f"   Random init - mean: {param_rand.mean():.6f}, std: {param_rand.std():.6f}")
    print(f"   Pretrained  - mean: {param_pre.mean():.6f}, std: {param_pre.std():.6f}")
    print(f"   Different: {not torch.allclose(param_rand, param_pre, atol=1e-6)}")

print(f"\n{'='*80}")
print("Comparison complete!")
print(f"{'='*80}\n")
