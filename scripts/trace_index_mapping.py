"""
Trace exactly what happens to indices during preprocessing
"""
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from pathlib import Path

# Import the actual transform
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from med3pipe.sam.transforms import ResizeLargestTo

# Find the GIST-013_CT.nii file
gist_path = Path("c:/Users/cahel/Desktop/Med3Tab-PFN")
ct_file = None

for file in gist_path.rglob("GIST-013_CT.nii*"):
    if "_embedding.pt" not in str(file):
        ct_file = file
        break

print(f"{'='*80}")
print(f"GIST-013 Preprocessing Index Mapping")
print(f"{'='*80}\n")

# Load original
nii = nib.load(ct_file)
original_data = nii.get_fdata()
print(f"ORIGINAL shape: {original_data.shape}")

# Create TorchIO subject and apply transforms
subject = tio.Subject(image=tio.ScalarImage(str(ct_file)))

print(f"\n{'='*80}")
print("Step-by-step transformation:")
print(f"{'='*80}\n")

# Step 1: ToCanonical
print("1. ToCanonical()")
subject = tio.ToCanonical()(subject)
shape_after_canonical = subject['image'].spatial_shape
print(f"   Shape: {shape_after_canonical}")

# Step 2: ResizeLargestTo(128)
print("\n2. ResizeLargestTo(128)")
max_dim = max(shape_after_canonical)
scale = 128 / max_dim
new_shape = tuple(int(dim * scale) for dim in shape_after_canonical)
print(f"   Max dimension: {max_dim}")
print(f"   Scale factor: {scale:.6f}")
print(f"   Target shape: {new_shape}")

subject = ResizeLargestTo(target_size=128)(subject)
shape_after_resize = subject['image'].spatial_shape
print(f"   Actual shape after resize: {shape_after_resize}")

# Step 3: CropOrPad
print("\n3. CropOrPad((128, 128, 128))")
# Calculate how much padding is needed in each dimension
pad_needed = [(128 - dim) for dim in shape_after_resize]
pad_before = [p // 2 for p in pad_needed]
pad_after = [p - pb for p, pb in zip(pad_needed, pad_before)]
print(f"   Padding needed per dimension: {pad_needed}")
print(f"   Padding before: {pad_before}")
print(f"   Padding after: {pad_after}")

subject = tio.CropOrPad(target_shape=(128, 128, 128))(subject)
shape_final = subject['image'].spatial_shape
print(f"   Final shape: {shape_final}")

# Get the final volume
final_volume = subject['image'].numpy()[0]

print(f"\n{'='*80}")
print("Index Mapping Analysis:")
print(f"{'='*80}\n")

# Map original indices to final indices
print("Sagittal (X) dimension mapping:")
print(f"  Original dimension: 0 to {original_data.shape[0]-1}")
print(f"  After resize: 0 to {shape_after_resize[0]-1}")
print(f"  After pad (with {pad_before[0]} padding before): {pad_before[0]} to {pad_before[0] + shape_after_resize[0] - 1}")
print(f"\n  MAPPING:")
for orig_x in [56, 68, 80, 100, 120]:
    # Map to resized space
    resized_x = orig_x * scale
    # Map to padded space
    final_x = resized_x + pad_before[0]
    
    status = ""
    if final_x < pad_before[0]:
        status = "→ in PADDING (before data)"
    elif final_x >= pad_before[0] + shape_after_resize[0]:
        status = "→ in PADDING (after data)"
    else:
        # Check if there's actual data
        final_x_int = int(round(final_x))
        if 0 <= final_x_int < 128:
            slice_data = final_volume[final_x_int, :, :]
            num_nonzero = np.count_nonzero(slice_data > -900)
            status = f"→ HAS DATA ({num_nonzero} non-air pixels)"
        else:
            status = "→ OUT OF BOUNDS"
    
    print(f"    Original X={orig_x:3d} → Resized X≈{resized_x:5.1f} → Final X≈{final_x:5.1f}  {status}")

# Check the actual data region boundaries
print(f"\n{'='*80}")
print("Data Region Boundaries:")
print(f"{'='*80}\n")
print(f"Resized data occupies: X={pad_before[0]} to X={pad_before[0] + shape_after_resize[0] - 1}")
print(f"\nSlices outside this range are padding!")

# Verify by checking actual slices
print(f"\n{'='*80}")
print("Verification - checking actual slices:")
print(f"{'='*80}\n")

for x_check in [10, 30, 56, 68, 80, 100, 120]:
    if 0 <= x_check < 128:
        slice_data = final_volume[x_check, :, :]
        num_unique = len(np.unique(slice_data))
        num_nonzero = np.count_nonzero(slice_data > -900)
        
        # Check if it's in the data region
        in_data_region = pad_before[0] <= x_check < pad_before[0] + shape_after_resize[0]
        
        print(f"X={x_check:3d}: unique_vals={num_unique:4d}, non-air_pixels={num_nonzero:5d}  {'[DATA REGION]' if in_data_region else '[PADDING REGION]'}")
