"""
Create a better visualization showing the data region and padding
"""
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from med3pipe.sam.transforms import ResizeLargestTo

# Try to import matplotlib, but continue without it if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Matplotlib not available, will only generate text output\n")

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

print(f"{'='*80}")
print("GIST-013 Visualization - Showing Data vs Padding Regions")
print(f"{'='*80}\n")

# Create better slice selections that show actual data
print("ISSUE FOUND:")
print("-" * 80)
print("The original GIST-013 volume is 512×512×76 (very thin in Z-axis)")
print("After preprocessing to 128³, the Z-dimension is heavily padded:")
print("  - Z=0 to Z=53: PADDING (54 slices)")
print("  - Z=54 to Z=72: REAL DATA (19 slices) ← actual anatomy here!")
print("  - Z=73 to Z=127: PADDING (55 slices)")
print()
print("When viewing sagittal slices (X=constant, Y vs Z plane):")
print("  - 85% of the view is GRAY PADDING (109/128 slices in Z)")
print("  - Only 15% shows real anatomy (19/128 slices in Z)")
print()
print("SOLUTION:")
print("For better visualization, use Z indices where data exists!")
print("Recommended Z values: 54, 60, 63, 66, 72")
print(f"{'='*80}\n")

# Show what happens at the three index pairs
print("Current visualization indices (Z=Y=X equal):")
print("-" * 80)
for idx in [56, 68, 80]:
    in_data_z = 54 <= idx <= 72
    sagittal_slice = final_volume[idx, :, :]  # X=idx, view Y vs Z
    
    # Count how much is data vs padding in Z
    data_region = sagittal_slice[:, 54:73]  # Only the Z slices with data
    pad_region = np.concatenate([sagittal_slice[:, :54], sagittal_slice[:, 73:]], axis=1)
    
    data_pixels = data_region.size
    pad_pixels = pad_region.size
    pad_fraction = pad_pixels / sagittal_slice.size
    
    print(f"\nAt Z=Y=X={idx}:")
    print(f"  Axial (Z={idx}):    {'DATA SLICE' if in_data_z else 'PADDING SLICE ⚠️'}")
    print(f"  Coronal (Y={idx}):  DATA SLICE (Y has no padding)")
    print(f"  Sagittal (X={idx}): {pad_fraction*100:.1f}% of view is padding")
    if pad_fraction > 0.8:
        print(f"                      ⚠️  Looks almost all gray!")

if HAS_MPL:
    print(f"\n{'='*80}")
    print("Creating visualization with data region highlighted...")
    print(f"{'='*80}\n")
    
    # Create a figure showing sagittal at X=56, 68, 80 with data region marked
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, ax in zip([56, 68, 80], axes):
        sagittal = final_volume[idx, :, :]  # Y vs Z
        
        # Display with proper windowing
        vmin, vmax = np.percentile(sagittal[sagittal > -900], [1, 99])
        im = ax.imshow(sagittal.T, cmap='gray', aspect='auto', 
                      origin='lower', vmin=vmin, vmax=vmax)
        
        # Highlight the data region in Z
        rect = patches.Rectangle((0, 54), 128, 19, 
                                linewidth=2, edgecolor='red', 
                                facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        ax.set_xlabel('Y', fontsize=12)
        ax.set_ylabel('Z', fontsize=12)
        ax.set_title(f'Sagittal X={idx}\n(Red box = actual data region Z=54-72)', 
                    fontsize=11)
        
        # Add text annotation
        ax.text(64, 10, 'PADDING', ha='center', va='center',
               color='red', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax.text(64, 63, 'DATA', ha='center', va='center',
               color='lime', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.text(64, 100, 'PADDING', ha='center', va='center',
               color='red', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    output_path = Path("c:/Users/cahel/Desktop/Med3Tab-PFN/scripts/gist013_padding_explanation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    print()
    
    # Create a second figure showing better indices
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    
    # Better indices that show actual data
    better_indices = [(56, 56, 60), (60, 60, 63), (68, 68, 66)]
    
    for row_idx, (z, y, x) in enumerate(better_indices):
        # Axial
        axes2[row_idx, 0].imshow(final_volume[:, :, z].T, cmap='gray', origin='lower')
        axes2[row_idx, 0].set_title(f'Axial Z={z}')
        axes2[row_idx, 0].set_xlabel('X')
        axes2[row_idx, 0].set_ylabel('Y')
        axes2[row_idx, 0].axhline(y, color='yellow', linewidth=1)
        axes2[row_idx, 0].axvline(x, color='yellow', linewidth=1)
        
        # Coronal
        axes2[row_idx, 1].imshow(final_volume[:, y, :].T, cmap='gray', origin='lower')
        axes2[row_idx, 1].set_title(f'Coronal Y={y}')
        axes2[row_idx, 1].set_xlabel('X')
        axes2[row_idx, 1].set_ylabel('Z')
        axes2[row_idx, 1].axhline(z, color='yellow', linewidth=1)
        axes2[row_idx, 1].axvline(x, color='yellow', linewidth=1)
        # Mark data region
        axes2[row_idx, 1].axhspan(54, 72, alpha=0.2, color='green')
        
        # Sagittal
        axes2[row_idx, 2].imshow(final_volume[x, :, :].T, cmap='gray', origin='lower')
        axes2[row_idx, 2].set_title(f'Sagittal X={x}')
        axes2[row_idx, 2].set_xlabel('Y')
        axes2[row_idx, 2].set_ylabel('Z')
        axes2[row_idx, 2].axhline(z, color='yellow', linewidth=1)
        axes2[row_idx, 2].axvline(y, color='yellow', linewidth=1)
        # Mark data region
        axes2[row_idx, 2].axhspan(54, 72, alpha=0.2, color='green')
    
    fig2.suptitle('GIST-013: Better Visualization (crosshairs at Z values with actual data)', 
                 fontsize=14, weight='bold')
    plt.tight_layout()
    output_path2 = Path("c:/Users/cahel/Desktop/Med3Tab-PFN/scripts/gist013_better_indices.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved improved visualization to: {output_path2}")
    plt.close('all')
    
print(f"\n{'='*80}")
print("RECOMMENDATION FOR YOUR NOTEBOOK:")
print(f"{'='*80}\n")
print("Instead of using equal indices (Z=Y=X), use indices that account for")
print("the data region boundaries. For GIST-013:")
print()
print("CURRENT (equal indices):")
print("  [(56, 56, 56), (68, 68, 68), (80, 80, 80)]")
print("  → Z=56,68,80 where only Z=56,68 have data; sagittal views are 85% padding")
print()
print("BETTER (Z in data region):")
print("  [(60, 60, 60), (63, 63, 63), (66, 66, 66)]")
print("  → All Z values in the data region [54-72]")
print()
print("OR detect data boundaries automatically:")
print("  - Find non-padding region in each dimension")
print("  - Select indices from within those regions")
print("  - This would make visualization work for ALL cases!")
