"""
Example: Running TabPFN with Lesion Size Filtering

This script demonstrates how to filter training/testing data based on
lesion size metrics (voxels, dimensions, density).
"""

from pathlib import Path
from med3pipe import run_single_dataset, LesionSizeFilter

# ==============================================================================
# Example 1: Basic Filtering - Voxels Only
# ==============================================================================

def example_1_voxels_only():
    """Filter by minimum voxel count only."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Filter by Voxel Count (>= 500 voxels)")
    print("="*70)
    
    result = run_single_dataset(
        method="tabpfn",
        dataset_root=Path("path/to/gist"),  # Change this
        category="gist",
        ct_name="ct_GIST",
        # Lesion filtering
        min_voxels=500,  # Only cases with >= 500 voxels
        # Other parameters
        split_ratio=0.8,
        seed=2025,
    )
    
    print("\n✅ Training complete!")
    print(f"Output directory: {result.tabpfn['out_dir']}")


# ==============================================================================
# Example 2: Recommended Multi-Criteria Filtering
# ==============================================================================

def example_2_recommended_filtering():
    """Use recommended filtering for viable SAM-Med3D cases."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Recommended Multi-Criteria Filtering")
    print("="*70)
    
    # Create filter with recommended thresholds
    lesion_filter = LesionSizeFilter(
        min_voxels=500,       # At least 500 voxels
        min_dimension=5,      # At least 5 voxels in smallest dimension
        min_density=0.3,      # At least 30% density (not too fragmented)
    )
    
    result = run_single_dataset(
        method="tabpfn",
        dataset_root=Path("path/to/gist"),  # Change this
        category="gist",
        ct_name="ct_GIST",
        lesion_filter=lesion_filter,
        split_ratio=0.8,
        seed=2025,
    )
    
    print("\n✅ Training complete!")
    print(f"Output directory: {result.tabpfn['out_dir']}")


# ==============================================================================
# Example 3: Conservative Filtering
# ==============================================================================

def example_3_conservative_filtering():
    """Use stricter filtering for highest quality cases."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Conservative (Strict) Filtering")
    print("="*70)
    
    result = run_single_dataset(
        method="tabpfn",
        dataset_root=Path("path/to/gist"),  # Change this
        category="gist",
        ct_name="ct_GIST",
        # Stricter thresholds
        min_voxels=1000,      # More voxels required
        min_dimension=10,     # Larger minimum dimension
        min_density=0.3,
        split_ratio=0.8,
        seed=2025,
    )
    
    print("\n✅ Training complete!")
    print(f"Output directory: {result.tabpfn['out_dir']}")


# ==============================================================================
# Example 4: Using LoCalPFN with Filtering
# ==============================================================================

def example_4_localpfn_with_filtering():
    """Run LoCalPFN with lesion filtering."""
    print("\n" + "="*70)
    print("EXAMPLE 4: LoCalPFN with Lesion Filtering")
    print("="*70)
    
    from med3pipe import LocalPFNConfig
    
    # Create LoCalPFN config
    local_cfg = LocalPFNConfig(
        n_neighbors=10,
        max_train_samples_per_class=100,
    )
    
    result = run_single_dataset(
        method="localpfn",  # Use LoCalPFN instead of TabPFN
        dataset_root=Path("path/to/gist"),  # Change this
        category="gist",
        ct_name="ct_GIST",
        # Lesion filtering
        min_voxels=500,
        min_dimension=5,
        # LoCalPFN config
        local_cfg=local_cfg,
        split_ratio=0.8,
        seed=2025,
    )
    
    print("\n✅ Training complete!")
    print(f"Output directory: {result.localpfn['out_dir']}")


# ==============================================================================
# Example 5: Custom Lesion CSV Path
# ==============================================================================

def example_5_custom_csv_path():
    """Specify custom path to lesion analysis CSV."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Lesion CSV Path")
    print("="*70)
    
    lesion_filter = LesionSizeFilter(
        min_voxels=500,
        lesion_csv_path=Path("custom/path/to/lesion_size_analysis.csv"),
    )
    
    result = run_single_dataset(
        method="tabpfn",
        dataset_root=Path("path/to/gist"),  # Change this
        category="gist",
        ct_name="ct_GIST",
        lesion_filter=lesion_filter,
    )
    
    print("\n✅ Training complete!")


# ==============================================================================
# Example 6: Check Filter Statistics Without Training
# ==============================================================================

def example_6_check_statistics():
    """Preview filtering statistics without running full pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Preview Filter Statistics")
    print("="*70)
    
    # Create filter
    lesion_filter = LesionSizeFilter(
        min_voxels=500,
        min_dimension=5,
        min_density=0.3,
    )
    
    # Print summary (shows which cases will be filtered)
    lesion_filter.print_filter_summary()
    
    # Get valid case IDs
    valid_cases = lesion_filter.get_valid_cases()
    print(f"\nValid cases (first 10): {sorted(list(valid_cases))[:10]}")
    print(f"Total valid cases: {len(valid_cases)}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║           Lesion Size Filtering Examples                             ║
    ║                                                                       ║
    ║  Demonstrates how to filter training/testing data based on           ║
    ║  lesion size metrics to focus on viable cases for SAM-Med3D.         ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    
    Prerequisites:
    1. Run: python scripts/analyze_lesion_sizes.py
       (Generates lesion_size_analysis.csv)
    
    2. Update dataset_root paths in the examples below
    
    Available metrics for filtering:
    - min_voxels: Minimum preprocessed voxel count (e.g., 500)
    - min_dimension: Minimum bounding box dimension (e.g., 5)
    - min_density: Minimum lesion density, 0-1 (e.g., 0.3)
    
    Note: "pct" in preprocessing_summary_per_dataset.csv is PERCENTAGE,
          not a filterable metric. Use the metrics above instead.
    """)
    
    # Uncomment the example you want to run:
    
    # example_1_voxels_only()
    # example_2_recommended_filtering()
    # example_3_conservative_filtering()
    # example_4_localpfn_with_filtering()
    # example_5_custom_csv_path()
    example_6_check_statistics()  # Safe to run - just shows stats
