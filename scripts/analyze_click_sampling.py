"""Analyze the click sampling logic to understand why refinement fails."""

import json
from pathlib import Path
import numpy as np

debug_file = Path("results/debug_logs/liver_Liver-001_MR_debug.json")

with open(debug_file) as f:
    data = json.load(f)

print("="*80)
print("Click Sampling Analysis")
print("="*80)
print(f"Case: {data['case']}")
print(f"GT voxels: {data['gt_voxels']}")
print(f"Final pred voxels: {data['pred_voxels']}")
print(f"Final Dice: {data['dice_score']}")
print()

per_click_dice = data['debug']['per_click_dice']
per_click_voxels = data['debug']['per_click_voxels']
per_click_labels = data['debug']['per_click_labels']
per_click_points = data['debug']['per_click_points']

gt_voxels = data['gt_voxels']

print(f"{'Click':<8} {'Dice':<10} {'Pred Vox':<12} {'Label':<8} {'FN Est':<12} {'FP Est':<12} {'Issue'}")
print("-"*95)

for i in range(len(per_click_dice)):
    click_num = i + 1
    dice = per_click_dice[i]
    pred_vox = per_click_voxels[i]
    label = "pos" if per_click_labels[i] == 1 else "neg"
    
    # Estimate FN and FP based on Dice formula
    # Dice = 2*TP / (2*TP + FP + FN)
    # If we know GT and Pred voxels, we can estimate:
    # TP (overlap) ≈ (Dice * (pred_vox + gt_voxels)) / 2
    if dice > 0:
        tp_estimate = (dice * (pred_vox + gt_voxels)) / 2
    else:
        tp_estimate = 0
    
    fn_estimate = gt_voxels - tp_estimate
    fp_estimate = pred_vox - tp_estimate
    
    # Identify issues
    issue = ""
    if fp_estimate > fn_estimate * 5:
        issue = "⚠️ FP >> FN (class imbalance)"
    if dice == per_click_dice[0] and i > 0:
        issue += " 🔴 No improvement"
    
    print(f"{click_num:<8} {dice:<10.4f} {pred_vox:<12} {label:<8} {fn_estimate:<12.0f} {fp_estimate:<12.0f} {issue}")

print("\n" + "="*80)
print("Key Insights")
print("="*80)

# Check if there's severe class imbalance
first_pred = per_click_voxels[0]
ratio = first_pred / gt_voxels
print(f"1. Initial prediction is {ratio:.1f}× larger than GT")
print(f"   → Prediction starts in wrong location with massive false positive region")

# Check label distribution
pos_clicks = sum(1 for label in per_click_labels if label == 1)
neg_clicks = sum(1 for label in per_click_labels if label == 0)
print(f"\n2. Click distribution: {pos_clicks} positive, {neg_clicks} negative")
print(f"   → Nearly balanced, but FP region is {per_click_voxels[-1] / gt_voxels:.1f}× larger than FN region")

# Check if Dice improves
dice_changes = [per_click_dice[i] - per_click_dice[i-1] for i in range(1, len(per_click_dice))]
improvements = sum(1 for d in dice_changes if d > 0)
print(f"\n3. Dice improvements: {improvements} out of {len(dice_changes)} clicks")
if improvements == 0:
    print("   ⚠️ ZERO improvement across all refinement clicks")

# Predicted voxel growth
voxel_growth = per_click_voxels[-1] - per_click_voxels[0]
print(f"\n4. Predicted mask grows by {voxel_growth} voxels ({voxel_growth/per_click_voxels[0]*100:.0f}% increase)")
print(f"   → Model keeps expanding prediction instead of localizing to GT")

print("\n" + "="*80)
print("Problem Diagnosis")
print("="*80)
print("""
The click sampling logic has a critical flaw:

Current behavior (line 215-221 in compute_sam_dice_scores.py):
  if len(fn_points) > 0 and len(fp_points) > 0:
      if np.random.random() > 0.5:  # ← 50/50 split
          point = fn_points[...]  # Sample from FN (helps localization)
      else:
          point = fp_points[...]  # Sample from FP (doesn't help when FP >> FN)

Issue:
- When prediction is spatially misaligned, FP region can be 10-100× larger than FN region
- 50/50 sampling wastes half the clicks on the massive FP region
- Negative clicks in wrong location don't help SAM find the correct target
- Model needs MORE positive clicks in FN region to discover the actual lesion location

Recommended fix:
  # Bias toward false negatives when there's severe class imbalance
  fn_count = len(fn_points)
  fp_count = len(fp_points)
  
  if fn_count > 0 and fp_count > 0:
      # If FP region is much larger, heavily bias toward FN
      if fp_count > fn_count * 3:
          prob_fn = 0.9  # 90% chance to sample FN
      else:
          prob_fn = 0.7  # 70% chance to sample FN (still favor finding target)
      
      if np.random.random() < prob_fn:
          point = fn_points[np.random.randint(fn_count)]
          is_positive = True
      else:
          point = fp_points[np.random.randint(fp_count)]
          is_positive = False
""")
