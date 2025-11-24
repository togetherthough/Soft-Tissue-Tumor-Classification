"""Analyze what happens with the first click to understand initialization failure."""

from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from importlib.machinery import SourceFileLoader
loader = SourceFileLoader('sam_dice_script', 'scripts/compute_sam_dice_scores.py')
mod = loader.load_module()

# Load the liver case
config_path = project_root / 'configs' / 'datasets_analysis.yaml'
datasets = mod.discover_cases_from_config(config_path, project_root)
liver_case = datasets['liver']['cases'][0]

print("="*80)
print("First Click Initialization Analysis")
print("="*80)
print(f"Case: {liver_case['case_id']}")

# Load and preprocess
image_sitk = mod.merge_images(liver_case['image_files'])
mask_sitk = mod.merge_segmentations(liver_case['segmentation_files'])

image_tensor = mod.load_volume_for_sam(image_sitk, img_size=128, modality='CT')
gt_mask = mod.load_mask_for_sam(mask_sitk, img_size=128)

print(f"\nPreprocessed shapes:")
print(f"  Image: {tuple(image_tensor.shape)}")
print(f"  GT mask: {gt_mask.shape}")
print(f"  GT voxels: {gt_mask.sum()}")

# Check GT mask distribution
coords = np.argwhere(gt_mask > 0)
if len(coords) > 0:
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    center_z = (z_min + z_max) // 2
    
    print(f"\nGT mask bounding box:")
    print(f"  X: [{x_min}, {x_max}] (size={x_max-x_min+1})")
    print(f"  Y: [{y_min}, {y_max}] (size={y_max-y_min+1})")
    print(f"  Z: [{z_min}, {z_max}] (size={z_max-z_min+1})")
    print(f"\nCenter point (first click location):")
    print(f"  (x={center_x}, y={center_y}, z={center_z})")
    
    # Check if center point is actually in the GT mask
    is_center_in_gt = gt_mask[center_z, center_y, center_x] > 0
    print(f"  Center point inside GT: {is_center_in_gt}")
    
    # Check GT mask density
    bbox_volume = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)
    density = gt_mask.sum() / bbox_volume if bbox_volume > 0 else 0
    print(f"\nGT mask density: {density:.2%} (voxels / bbox volume)")
    
    # Show ASCII visualization of center slices
    print("\nASCII visualization of GT mask at center:")
    print("Axial slice (XY plane at z={})".format(center_z))
    axial_slice = gt_mask[center_z, max(0,center_y-10):center_y+11, max(0,center_x-10):center_x+11]
    for i in range(axial_slice.shape[0]):
        row = ''
        for j in range(axial_slice.shape[1]):
            if i == 10 and j == 10:
                row += '+'  # Center click
            elif axial_slice[i, j] > 0:
                row += '#'
            else:
                row += '.'
        print(row)
    
    # Check surrounding voxels
    window_size = 5
    x_slice = slice(max(0, center_x-window_size), min(128, center_x+window_size+1))
    y_slice = slice(max(0, center_y-window_size), min(128, center_y+window_size+1))
    z_slice = slice(max(0, center_z-window_size), min(128, center_z+window_size+1))
    
    local_gt = gt_mask[z_slice, y_slice, x_slice]
    local_voxels = local_gt.sum()
    local_total = local_gt.size
    
    print(f"\nLocal neighborhood (±{window_size} voxels around center):")
    print(f"  GT voxels: {local_voxels} / {local_total} ({local_voxels/local_total*100:.1f}%)")
    
else:
    print("\n❌ GT mask is empty!")

print("\n" + "="*80)
print("Diagnosis:")
print("="*80)
print("""
If the GT mask is very sparse (low density) or the center point is not
actually inside the GT region, the first click won't provide a good
initialization for SAM. The model might activate on surrounding tissue
rather than the lesion itself.

Possible issues:
1. GT mask too small after downsampling (< 1% of bbox volume)
2. Center point landing outside GT due to sparse/disconnected regions
3. Image context around GT lesion looks similar to other anatomical structures
4. SAM's image encoder not trained on such small, downsampled targets
""")
