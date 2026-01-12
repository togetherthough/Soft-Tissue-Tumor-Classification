"""Generate per-dataset summary of preprocessed lesion shapes."""

from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
lesion_csv = project_root / 'results' / 'lesion_size_analysis.csv'

if not lesion_csv.exists():
    print(f"❌ Run analyze_lesion_sizes.py first to generate {lesion_csv}")
    exit(1)

df = pd.read_csv(lesion_csv)

print("="*80)
print("Per-Dataset Lesion Shape Summary (After 128³ Preprocessing)")
print("="*80)

summaries = []

for dataset in sorted(df['dataset'].unique()):
    ds_df = df[df['dataset'] == dataset]
    
    # Filter out empty cases
    non_empty = ds_df[ds_df['preproc_voxels'] > 0]
    
    summary = {
        'dataset': dataset,
        'total_cases': len(ds_df),
        'empty_cases': len(ds_df[ds_df['preproc_voxels'] == 0]),
        'valid_cases': len(non_empty),
        
        # Voxel statistics
        'mean_voxels': non_empty['preproc_voxels'].mean(),
        'median_voxels': non_empty['preproc_voxels'].median(),
        'std_voxels': non_empty['preproc_voxels'].std(),
        'min_voxels': non_empty['preproc_voxels'].min(),
        'max_voxels': non_empty['preproc_voxels'].max(),
        'q25_voxels': non_empty['preproc_voxels'].quantile(0.25),
        'q75_voxels': non_empty['preproc_voxels'].quantile(0.75),
        
        # Bounding box statistics (mean dimensions)
        'mean_bbox_x': non_empty['bbox_x'].mean(),
        'mean_bbox_y': non_empty['bbox_y'].mean(),
        'mean_bbox_z': non_empty['bbox_z'].mean(),
        'median_bbox_x': non_empty['bbox_x'].median(),
        'median_bbox_y': non_empty['bbox_y'].median(),
        'median_bbox_z': non_empty['bbox_z'].median(),
        
        # Minimum dimension statistics
        'mean_min_dim': non_empty['min_dimension'].mean(),
        'median_min_dim': non_empty['min_dimension'].median(),
        'min_dim_under_5': len(non_empty[non_empty['min_dimension'] < 5]),
        
        # Density statistics
        'mean_density': non_empty['density'].mean(),
        'median_density': non_empty['density'].median(),
        
        # Downsampling ratio
        'mean_downsample_ratio': non_empty['downsampling_ratio'].mean(),
        'median_downsample_ratio': non_empty['downsampling_ratio'].median(),
        
        # Filtering statistics
        'cases_gte_500_voxels': len(non_empty[non_empty['preproc_voxels'] >= 500]),
        'cases_gte_1000_voxels': len(non_empty[non_empty['preproc_voxels'] >= 1000]),
        'cases_min_dim_gte_5': len(non_empty[non_empty['min_dimension'] >= 5]),
        'cases_viable': len(non_empty[(non_empty['preproc_voxels'] >= 500) & 
                                       (non_empty['min_dimension'] >= 5)]),
    }
    
    summaries.append(summary)
    
    # Print human-readable summary
    print(f"\n{dataset.upper()}")
    print("-" * 80)
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  Empty after preprocessing: {summary['empty_cases']}")
    print(f"  Valid cases: {summary['valid_cases']}")
    print(f"\n  Preprocessed Voxels:")
    print(f"    Mean: {summary['mean_voxels']:.0f} ± {summary['std_voxels']:.0f}")
    print(f"    Median: {summary['median_voxels']:.0f}")
    print(f"    Range: [{summary['min_voxels']:.0f}, {summary['max_voxels']:.0f}]")
    print(f"    Q25-Q75: [{summary['q25_voxels']:.0f}, {summary['q75_voxels']:.0f}]")
    print(f"\n  Mean Bounding Box Dimensions (voxels):")
    print(f"    X: {summary['mean_bbox_x']:.1f} (median: {summary['median_bbox_x']:.0f})")
    print(f"    Y: {summary['mean_bbox_y']:.1f} (median: {summary['median_bbox_y']:.0f})")
    print(f"    Z: {summary['mean_bbox_z']:.1f} (median: {summary['median_bbox_z']:.0f})")
    print(f"\n  Minimum Dimension:")
    print(f"    Mean: {summary['mean_min_dim']:.1f}")
    print(f"    Median: {summary['median_min_dim']:.0f}")
    print(f"    Cases with min_dim < 5: {summary['min_dim_under_5']} ({summary['min_dim_under_5']/summary['valid_cases']*100:.1f}%)")
    print(f"\n  Lesion Density (voxels/bbox_volume):")
    print(f"    Mean: {summary['mean_density']:.2%}")
    print(f"    Median: {summary['median_density']:.2%}")
    print(f"\n  Downsampling Ratio (raw/preprocessed):")
    print(f"    Mean: {summary['mean_downsample_ratio']:.1f}×")
    print(f"    Median: {summary['median_downsample_ratio']:.1f}×")
    print(f"\n  Viable Cases (≥500 voxels AND min_dim≥5):")
    print(f"    Count: {summary['cases_viable']} ({summary['cases_viable']/summary['valid_cases']*100:.1f}%)")

# Create summary DataFrame
summary_df = pd.DataFrame(summaries)

# Round numerical columns for readability
numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if 'ratio' in col or 'density' in col:
        summary_df[col] = summary_df[col].round(3)
    else:
        summary_df[col] = summary_df[col].round(1)

# Save to data folder
output_path = project_root / 'data' / 'preprocessed_lesion_summary.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(output_path, index=False)

print("\n" + "="*80)
print(f"✅ Saved per-dataset summary to: {output_path}")

# Also create an overall summary
print("\n" + "="*80)
print("OVERALL SUMMARY (All Datasets)")
print("="*80)
print(f"Total cases: {len(df)}")
print(f"Empty cases: {len(df[df['preproc_voxels'] == 0])}")
print(f"Valid cases: {len(df[df['preproc_voxels'] > 0])}")
print(f"\nCases meeting viability criteria:")
print(f"  ≥500 voxels: {len(df[df['preproc_voxels'] >= 500])} ({len(df[df['preproc_voxels'] >= 500])/len(df)*100:.1f}%)")
print(f"  min_dim ≥5: {len(df[df['min_dimension'] >= 5])} ({len(df[df['min_dimension'] >= 5])/len(df)*100:.1f}%)")
viable = df[(df['preproc_voxels'] >= 500) & (df['min_dimension'] >= 5)]
print(f"  Both criteria: {len(viable)} ({len(viable)/len(df)*100:.1f}%)")

# Show distribution by viability tier
print("\n" + "-"*80)
print("Viability Tiers:")
print("-"*80)
tiers = [
    ("Tier 1: Very Small (< 100 voxels)", df['preproc_voxels'] < 100),
    ("Tier 2: Small (100-500 voxels)", (df['preproc_voxels'] >= 100) & (df['preproc_voxels'] < 500)),
    ("Tier 3: Medium (500-2000 voxels)", (df['preproc_voxels'] >= 500) & (df['preproc_voxels'] < 2000)),
    ("Tier 4: Large (≥2000 voxels)", df['preproc_voxels'] >= 2000),
]
for tier_name, mask in tiers:
    count = mask.sum()
    pct = count / len(df) * 100
    print(f"{tier_name:<35} {count:>4} cases ({pct:>5.1f}%)")
