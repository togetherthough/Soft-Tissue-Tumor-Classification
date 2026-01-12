"""
Debug script to check volume dimensions and padding for GIST-013_CT.nii
"""
import nibabel as nib
import numpy as np
from pathlib import Path

# Find the GIST-013_CT.nii file
gist_path = Path("c:/Users/cahel/Desktop/Med3Tab-PFN")

# Search for the file
possible_paths = [
    gist_path / "data" / "GIST" / "GIST-013_CT.nii",
    gist_path / "data" / "GIST" / "GIST-013_CT.nii.gz",
]

ct_file = None
for path in possible_paths:
    if path.exists():
        ct_file = path
        break

if ct_file is None:
    # Try to find it recursively
    print("Searching for GIST-013_CT.nii...")
    for file in gist_path.rglob("GIST-013_CT.nii*"):
        # Skip embedding files
        if "_embedding.pt" not in str(file):
            ct_file = file
            print(f"Found: {ct_file}")
            break

if ct_file is None:
    print("ERROR: Could not find GIST-013_CT.nii file")
    exit(1)

print(f"\n{'='*80}")
print(f"Loading: {ct_file}")
print(f"{'='*80}\n")

# Load the volume
nii = nib.load(ct_file)
ct_volume = nii.get_fdata()

print(f"Original CT volume shape: {ct_volume.shape}")
print(f"Original CT volume min: {ct_volume.min():.2f}, max: {ct_volume.max():.2f}")
print(f"Data type: {ct_volume.dtype}")

# Check non-zero content in each dimension
non_zero_x = np.any(ct_volume != ct_volume[0, 0, 0], axis=(1, 2))
non_zero_y = np.any(ct_volume != ct_volume[0, 0, 0], axis=(0, 2))
non_zero_z = np.any(ct_volume != ct_volume[0, 0, 0], axis=(0, 1))

print(f"\nNon-background slices:")
print(f"  X-axis: {non_zero_x.sum()} / {ct_volume.shape[0]} slices have content")
print(f"  Y-axis: {non_zero_y.sum()} / {ct_volume.shape[1]} slices have content")
print(f"  Z-axis: {non_zero_z.sum()} / {ct_volume.shape[2]} slices have content")

print(f"\nContent range:")
print(f"  X-axis: indices {np.where(non_zero_x)[0].min()} to {np.where(non_zero_x)[0].max()}")
print(f"  Y-axis: indices {np.where(non_zero_y)[0].min()} to {np.where(non_zero_y)[0].max()}")
print(f"  Z-axis: indices {np.where(non_zero_z)[0].min()} to {np.where(non_zero_z)[0].max()}")

# Check specific slices at X=56, X=68, X=80
print(f"\n{'='*80}")
print("Checking specific sagittal slices:")
print(f"{'='*80}\n")

for x_idx in [56, 68, 80]:
    if x_idx < ct_volume.shape[0]:
        slice_data = ct_volume[x_idx, :, :]
        unique_vals = np.unique(slice_data)
        has_content = len(unique_vals) > 1 or (len(unique_vals) == 1 and unique_vals[0] != ct_volume[0, 0, 0])
        print(f"X={x_idx}:")
        print(f"  Shape: {slice_data.shape}")
        print(f"  Unique values: {len(unique_vals)}")
        print(f"  Min: {slice_data.min():.2f}, Max: {slice_data.max():.2f}")
        print(f"  Has content: {has_content}")
        print(f"  Non-zero pixels: {np.count_nonzero(slice_data != ct_volume[0, 0, 0])}")
        print()
    else:
        print(f"X={x_idx}: OUT OF BOUNDS (volume shape is {ct_volume.shape})")
        print()

# Skip visualization for now (matplotlib not available)

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
if 80 < ct_volume.shape[0]:
    if non_zero_x[80]:
        print(f"✓ X=80 is within bounds and has content")
    else:
        print(f"⚠ X=80 is within bounds but appears to be padding/background")
else:
    print(f"✗ X=80 is OUT OF BOUNDS (volume only goes up to X={ct_volume.shape[0]-1})")
    print(f"  The visualization notebook is using indices beyond the actual volume!")
