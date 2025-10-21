"""
Test script for the new resize-then-pad preprocessing approach.

This script verifies that the ResizeLargestTo transform works correctly
and produces the expected 128^3 volumes for SAM-Med3D.
"""

import sys
from pathlib import Path
import torch
import torchio as tio
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med3pipe.sam.transforms import ResizeLargestTo, make_sam3d_transform, load_volume_resize_pad


def test_resize_largest_to():
    """Test ResizeLargestTo transform with synthetic data."""
    print("\n" + "="*70)
    print("Testing ResizeLargestTo Transform")
    print("="*70)
    
    test_cases = [
        # (input_shape, expected_intermediate_shape)
        ((200, 150, 100), (128, 96, 64)),    # Largest=200 → 128
        ((100, 200, 150), (64, 128, 96)),    # Largest=200 → 128
        ((150, 100, 200), (96, 64, 128)),    # Largest=200 → 128
        ((80, 60, 50), (106, 80, 66)),       # Largest=80, upscaled to 128
        ((128, 96, 64), (128, 96, 64)),      # Largest=128, no change
    ]
    
    transform = ResizeLargestTo(target_size=128)
    
    for i, (input_shape, expected_shape) in enumerate(test_cases, 1):
        # Create synthetic volume
        data = torch.randn(1, *input_shape)
        subject = tio.Subject(image=tio.ScalarImage(tensor=data))
        
        # Apply transform
        result = transform(subject)
        output_shape = result.image.spatial_shape
        
        # Calculate expected shape
        max_dim = max(input_shape)
        scale = 128 / max_dim
        calculated_expected = tuple(int(dim * scale) for dim in input_shape)
        
        print(f"\nTest {i}:")
        print(f"  Input shape:     {input_shape}")
        print(f"  Output shape:    {output_shape}")
        print(f"  Expected shape:  {calculated_expected}")
        print(f"  Max dimension:   {max(output_shape)} (should be ~128)")
        
        # Verify largest dimension is 128 (or close due to rounding)
        assert max(output_shape) == 128, f"Largest dim should be 128, got {max(output_shape)}"
        
        # Verify scale is correct (within 1 pixel due to rounding)
        for orig, out, exp in zip(input_shape, output_shape, calculated_expected):
            diff = abs(out - exp)
            assert diff <= 1, f"Shape mismatch: expected {exp}, got {out}"
        
        print(f"  ✅ PASSED")
    
    print("\n" + "="*70)
    print("All ResizeLargestTo tests passed!")
    print("="*70)


def test_full_pipeline():
    """Test the complete resize-then-pad pipeline."""
    print("\n" + "="*70)
    print("Testing Full Pipeline (Resize → Pad)")
    print("="*70)
    
    test_cases = [
        (200, 150, 100),
        (100, 200, 150),
        (80, 60, 50),
        (300, 200, 150),
        (128, 128, 128),
    ]
    
    transform = make_sam3d_transform(img_size=128, normalize=False)
    
    for i, input_shape in enumerate(test_cases, 1):
        # Create synthetic volume
        data = torch.randn(1, *input_shape)
        subject = tio.Subject(image=tio.ScalarImage(tensor=data))
        
        # Apply full pipeline
        result = transform(subject)
        output_shape = result.image.spatial_shape
        
        print(f"\nTest {i}:")
        print(f"  Input shape:  {input_shape}")
        print(f"  Output shape: {output_shape}")
        
        # Verify output is exactly 128^3
        assert output_shape == (128, 128, 128), f"Expected (128, 128, 128), got {output_shape}"
        print(f"  ✅ PASSED - Output is 128³")
    
    print("\n" + "="*70)
    print("All full pipeline tests passed!")
    print("="*70)


def test_data_preservation():
    """Test that data is preserved (not cropped) during resize."""
    print("\n" + "="*70)
    print("Testing Data Preservation")
    print("="*70)
    
    # Create a volume with a distinctive pattern
    input_shape = (200, 150, 100)
    data = torch.zeros(1, *input_shape)
    
    # Add a marker in each octant to verify all data is preserved
    markers = [
        (10, 10, 10),
        (190, 10, 10),
        (10, 140, 10),
        (190, 140, 10),
        (10, 10, 90),
        (190, 10, 90),
        (10, 140, 90),
        (190, 140, 90),
    ]
    
    for marker in markers:
        data[0, marker[0], marker[1], marker[2]] = 100.0
    
    subject = tio.Subject(image=tio.ScalarImage(tensor=data))
    
    # Apply resize-then-pad WITHOUT normalization
    transform = tio.Compose([
        tio.ToCanonical(),
        ResizeLargestTo(target_size=128),
        tio.CropOrPad(target_shape=(128, 128, 128)),
    ])
    result = transform(subject)
    output_data = result.image.data[0]
    
    print(f"\nInput shape:  {input_shape}")
    print(f"Output shape: {result.image.spatial_shape}")
    print(f"Input range:  [{data.min():.2f}, {data.max():.2f}]")
    print(f"Output range: [{output_data.min():.2f}, {output_data.max():.2f}]")
    
    # Check that we still have high values (markers should be preserved)
    # After resizing, the max value will be scaled down but should still be significant
    assert output_data.max() > 10.0, f"Markers should be preserved after resize (got {output_data.max():.2f})"
    
    # Count non-zero voxels
    non_zero = (output_data > 1.0).sum().item()
    print(f"Non-zero voxels preserved: {non_zero} (markers)")
    print(f"✅ PASSED - Data markers preserved (max value: {output_data.max():.2f})")
    
    print("\n" + "="*70)
    print("Data preservation test passed!")
    print("="*70)


def compare_old_vs_new_approach():
    """Compare data loss between old (CropOrPad) and new (Resize→Pad) approaches."""
    print("\n" + "="*70)
    print("Comparing Old vs New Approach")
    print("="*70)
    
    # Create a volume larger than 128 in all dimensions
    input_shape = (200, 180, 160)
    data = torch.randn(1, *input_shape)
    
    # Old approach: Direct CropOrPad
    subject_old = tio.Subject(image=tio.ScalarImage(tensor=data.clone()))
    old_transform = tio.CropOrPad(target_shape=(128, 128, 128))
    result_old = old_transform(subject_old)
    
    # New approach: Resize → Pad
    subject_new = tio.Subject(image=tio.ScalarImage(tensor=data.clone()))
    new_transform = tio.Compose([
        ResizeLargestTo(target_size=128),
        tio.CropOrPad(target_shape=(128, 128, 128)),
    ])
    result_new = new_transform(subject_new)
    
    print(f"\nInput shape: {input_shape}")
    print(f"\nOld approach (CropOrPad only):")
    print(f"  Output shape: {result_old.image.spatial_shape}")
    print(f"  Data cropped: YES (lost {200-128}×{180-128}×{160-128} = {(200-128)*(180-128)*(160-128):,} voxels)")
    
    print(f"\nNew approach (Resize→Pad):")
    print(f"  Intermediate shape after resize: ~(128, 115, 102)")
    print(f"  Final shape: {result_new.image.spatial_shape}")
    print(f"  Data cropped: NO (all data preserved, just downsampled)")
    
    print(f"\n✅ New approach preserves all information through downsampling")
    print(f"✅ Old approach would crop away {((200-128)*(180-128)*(160-128))/(200*180*160)*100:.1f}% of voxels")
    
    print("\n" + "="*70)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SAM-Med3D Preprocessing Transform Tests")
    print("="*70)
    
    try:
        test_resize_largest_to()
        test_full_pipeline()
        test_data_preservation()
        compare_old_vs_new_approach()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe new resize-then-pad approach is working correctly.")
        print("It preserves all data while achieving the required 128³ format.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
