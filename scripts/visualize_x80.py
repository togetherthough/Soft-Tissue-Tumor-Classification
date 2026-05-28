"""
Visualize what's actually at X=80 in the preprocessed volume
"""
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from med3pipe.sam.transforms import ResizeLargestTo

# Find and load GIST-013
gist_path = Path("c:/Users/cahel/Desktop/Med3Tab-PFN")
for file in gist_path.rglob("GIST-013_CT.nii*"):
    if "_embedding.pt" not in str(file):
        ct_file = file
        break

# Apply preprocessing
subject = tio.Subject(image=tio.ScalarImage(str(ct_file)))
transform = tio.Compose([
    tio.ToCanonical(),
    ResizeLargestTo(target_size=128),
    tio.CropOrPad(target_shape=(128, 128, 128)),
])
subject = transform(subject)
final_volume = subject['image'].numpy()[0]

print(f"Final volume shape: {final_volume.shape}")
print(f"\nAnalyzing sagittal slice at X=80:")
print(f"{'='*80}\n")

slice_x80 = final_volume[80, :, :]
print(f"Slice shape: {slice_x80.shape}")
print(f"Min value: {slice_x80.min():.2f}")
print(f"Max value: {slice_x80.max():.2f}")
print(f"Mean value: {slice_x80.mean():.2f}")
print(f"Std value: {slice_x80.std():.2f}")

unique_vals = np.unique(slice_x80)
print(f"\nUnique values: {len(unique_vals)}")
print(f"First 10 unique values: {unique_vals[:10]}")

# Check if it's mostly one value (padding)
most_common_val = np.bincount(slice_x80.flatten().astype(int) + 1000).argmax() - 1000
count_most_common = np.sum(np.abs(slice_x80 - most_common_val) < 0.1)
fraction_most_common = count_most_common / slice_x80.size

print(f"\nMost common value: {most_common_val:.2f}")
print(f"Fraction of pixels with this value: {fraction_most_common:.3f}")

# Check if it looks like padding
is_mostly_uniform = fraction_most_common > 0.9
print(f"\nIs this slice mostly uniform (>90% same value)? {is_mostly_uniform}")

if is_mostly_uniform:
    print("⚠️  THIS SLICE APPEARS TO BE PADDING!")
else:
    print("✓ This slice has varied content")

# Check where the data actually is
print(f"\n{'='*80}")
print("Checking data distribution across Z dimension (axial slices):")
print(f"{'='*80}\n")

# The Z dimension was heavily padded (from 19 to 128)
# Padding was [0, 0, 54] before and [0, 0, 55] after
# So actual data is at Z=54 to Z=72 (19 slices)

print("Z-dimension analysis:")
for z_idx in [0, 20, 40, 54, 60, 70, 72, 90, 110, 127]:
    z_slice = final_volume[:, :, z_idx]
    unique_z = len(np.unique(z_slice))
    nonzero_z = np.count_nonzero(z_slice > -900)
    in_data_range = 54 <= z_idx <= 72
    print(f"  Z={z_idx:3d}: unique={unique_z:4d}, non-air={nonzero_z:5d}  {'[DATA]' if in_data_range else '[PAD]'}")

print(f"\n{'='*80}")
print("Now checking the X=80, Z=? slice:")
print(f"{'='*80}\n")

# Check specific Z slices at X=80
print("At X=80, checking various Z positions:")
for z_idx in [40, 54, 60, 70, 72, 80]:
    if z_idx < final_volume.shape[2]:
        pixel_val = slice_x80[:, z_idx]
        unique_in_row = len(np.unique(pixel_val))
        mean_val = pixel_val.mean()
        in_data_z = 54 <= z_idx <= 72
        print(f"  Z={z_idx:3d}: unique_vals={unique_in_row:3d}, mean={mean_val:7.2f}  {'[DATA Z]' if in_data_z else '[PAD Z]'}")

# Show the actual pixel values at X=80, Y=80, various Z
print(f"\nPixel values at (X=80, Y=80, Z=...):")
for z_idx in [40, 54, 60, 70, 72, 80, 90]:
    val = final_volume[80, 80, z_idx]
    in_data_z = 54 <= z_idx <= 72
    print(f"  (80, 80, {z_idx:3d}) = {val:8.2f}  {'[DATA Z]' if in_data_z else '[PAD Z]'}")

# The key insight: X and Y are fully covered (no padding), but Z has padding
print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}\n")
print("Original volume: 512×512×76")
print("After resize: 128×128×19")
print("After pad: 128×128×128")
print()
print("Padding distribution:")
print("  X: NO padding (0 before, 0 after)")
print("  Y: NO padding (0 before, 0 after)")
print("  Z: HEAVY padding (54 before, 55 after)")
print()
print("Data region:")
print("  X: 0 to 127 (all slices have data)")
print("  Y: 0 to 127 (all slices have data)")
print("  Z: 54 to 72 (only 19 slices have data!)")
print()
print("At X=80:")
print("  - The sagittal slice DOES intersect real anatomy in X and Y")
print("  - BUT most of the Z dimension (axial) is PADDING")
print("  - When viewing X=80 as a 2D slice (Y vs Z), you see:")
print("    * Y=0 to Y=127: real anatomy")
print("    * Z=0 to Z=53: PADDING (gray)")
print("    * Z=54 to Z=72: REAL DATA")
print("    * Z=73 to Z=127: PADDING (gray)")
print()
if is_mostly_uniform:
    print("⚠️  The sagittal view at X=80 looks all gray because")
    print("    the Z dimension is 85% padding (109 out of 128 slices)!")
