"""
Debug script to understand how GIST-013 gets preprocessed from 512x512x76 to 128x128x128
and why X=80 ends up being all padding.
"""
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from pathlib import Path
import SimpleITK as sitk

# Custom ResizeLargestTo transform (copied from notebook pattern)
class ResizeLargestTo(tio.Transform):
    """Resize largest dimension to target_size, keeping aspect ratio."""
    def __init__(self, target_size, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size
    
    def apply_transform(self, subject):
        img = subject['image']
        original_shape = img.shape[1:]  # (D, H, W)
        max_dim = max(original_shape)
        scale = self.target_size / max_dim
        
        new_shape = tuple(int(round(dim * scale)) for dim in original_shape)
        print(f"    ResizeLargestTo: {original_shape} -> {new_shape} (scale={scale:.4f})")
        
        subject = tio.Resample(target=tuple(1/scale for _ in range(3)))(subject)
        return subject

# Find the GIST-013_CT.nii file
gist_path = Path("c:/Users/cahel/Desktop/Med3Tab-PFN")
ct_file = None

for file in gist_path.rglob("GIST-013_CT.nii*"):
    if "_embedding.pt" not in str(file):
        ct_file = file
        break

if ct_file is None:
    print("ERROR: Could not find GIST-013_CT.nii file")
    exit(1)

print(f"{'='*80}")
print(f"Loading: {ct_file}")
print(f"{'='*80}\n")

# Load the original volume
nii = nib.load(ct_file)
ct_volume = nii.get_fdata()

print(f"ORIGINAL volume shape: {ct_volume.shape}")
print(f"  X-axis (sagittal): 0 to {ct_volume.shape[0]-1}")
print(f"  Y-axis (coronal):  0 to {ct_volume.shape[1]-1}")
print(f"  Z-axis (axial):    0 to {ct_volume.shape[2]-1}")

# Check which slices have content in original volume
has_content_x = []
for x in [56, 68, 80]:
    if x < ct_volume.shape[0]:
        slice_data = ct_volume[x, :, :]
        non_background = np.count_nonzero(slice_data > -900)  # not air
        has_content_x.append((x, non_background > 100))
        print(f"  Original X={x}: {non_background} non-air pixels -> {'HAS CONTENT' if non_background > 100 else 'PADDING'}")

print(f"\n{'='*80}")
print("Applying preprocessing pipeline (like notebook)")
print(f"{'='*80}\n")

# Create TorchIO subject
subject = tio.Subject(
    image=tio.ScalarImage(str(ct_file))
)

# Apply transforms
img_size = 128

print("Step 1: ToCanonical()")
subject = tio.ToCanonical()(subject)
print(f"  After ToCanonical: {subject['image'].shape[1:]}")

print("\nStep 2: ResizeLargestTo(128)")
transform_resize = ResizeLargestTo(target_size=img_size)
subject = transform_resize(subject)
after_resize_shape = subject['image'].shape[1:]
print(f"  After Resize: {after_resize_shape}")

print("\nStep 3: CropOrPad((128, 128, 128))")
transform_pad = tio.CropOrPad(target_shape=(img_size, img_size, img_size))
subject = transform_pad(subject)
final_shape = subject['image'].shape[1:]
print(f"  After CropOrPad: {final_shape}")

# Get final volume
final_volume = subject['image'].numpy()[0]

print(f"\n{'='*80}")
print("FINAL PREPROCESSED volume:")
print(f"{'='*80}\n")
print(f"Shape: {final_volume.shape}")

# Check which slices have content in final volume
print("\nChecking sagittal slices in preprocessed volume:")
for x in [56, 68, 80]:
    if x < final_volume.shape[0]:
        slice_data = final_volume[x, :, :]
        unique_vals = np.unique(slice_data)
        non_background = np.count_nonzero(slice_data > -900)
        print(f"  Preprocessed X={x}:")
        print(f"    Unique values: {len(unique_vals)}")
        print(f"    Non-air pixels: {non_background}")
        print(f"    Status: {'HAS CONTENT' if non_background > 100 else '⚠️  ALL PADDING/AIR'}")

print(f"\n{'='*80}")
print("ANALYSIS:")
print(f"{'='*80}\n")

# Calculate what happened during resize
original_x_dim = ct_volume.shape[0]  # 512
final_x_dim = final_volume.shape[0]  # 128
scale_factor = final_x_dim / original_x_dim

print(f"Original X dimension: {original_x_dim}")
print(f"After resize, X dimension: {after_resize_shape[0]}")
print(f"After pad, X dimension: {final_x_dim}")
print(f"\nScale factor: {scale_factor:.4f}")
print(f"\nMapping from original to preprocessed:")
for orig_x in [56, 68, 80]:
    scaled_x = orig_x * scale_factor
    print(f"  Original X={orig_x} -> Preprocessed X≈{scaled_x:.1f}")

# Show what the actual data region is in preprocessed space
original_data_end = original_x_dim  # All 512 slices have content
scaled_data_end = original_data_end * scale_factor
print(f"\nOriginal data region: X=0 to X={original_x_dim-1}")
print(f"Preprocessed data region (scaled): X=0 to X≈{scaled_data_end:.1f}")

if after_resize_shape[0] < img_size:
    pad_amount = (img_size - after_resize_shape[0]) / 2
    print(f"\nPadding added: {pad_amount:.1f} on each side")
    print(f"Actual data in preprocessed volume: X≈{pad_amount:.1f} to X≈{pad_amount + after_resize_shape[0]:.1f}")
