"""Analyze preprocessed lesion sizes to determine viable cases for SAM-Med3D."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from importlib.machinery import SourceFileLoader
loader = SourceFileLoader('sam_dice_script', 'scripts/compute_sam_dice_scores.py')
mod = loader.load_module()

config_path = project_root / 'configs' / 'datasets_analysis.yaml'
print("Discovering datasets...")
datasets = mod.discover_cases_from_config(config_path, project_root)

results = []

print("\nAnalyzing preprocessed lesion sizes...")
print("="*80)

for dataset_name, info in datasets.items():
    modality = info.get('modality', 'CT')
    print(f"\n📊 Dataset: {dataset_name} ({len(info['cases'])} cases)")
    
    for case in tqdm(info['cases'], desc=f"  {dataset_name}", unit="case"):
        try:
            # Load and preprocess
            image_sitk = mod.merge_images(case['image_files'])
            mask_sitk = mod.merge_segmentations(case['segmentation_files'])
            
            # Get raw sizes
            raw_mask = np.array(mod.sitk.GetArrayFromImage(mask_sitk))
            raw_voxels = (raw_mask > 0).sum()
            
            # Get preprocessed sizes
            gt_mask = mod.load_mask_for_sam(mask_sitk, img_size=128)
            preproc_voxels = gt_mask.sum()
            
            # Compute bounding box dimensions
            coords = np.argwhere(gt_mask > 0)
            if len(coords) > 0:
                z_min, y_min, x_min = coords.min(axis=0)
                z_max, y_max, x_max = coords.max(axis=0)
                bbox_x = x_max - x_min + 1
                bbox_y = y_max - y_min + 1
                bbox_z = z_max - z_min + 1
                bbox_volume = bbox_x * bbox_y * bbox_z
                density = preproc_voxels / bbox_volume if bbox_volume > 0 else 0
                
                # Minimum dimension
                min_dim = min(bbox_x, bbox_y, bbox_z)
            else:
                bbox_x = bbox_y = bbox_z = bbox_volume = density = min_dim = 0
            
            results.append({
                'dataset': dataset_name,
                'case': case['case_id'],
                'raw_voxels': int(raw_voxels),
                'preproc_voxels': int(preproc_voxels),
                'downsampling_ratio': raw_voxels / preproc_voxels if preproc_voxels > 0 else np.inf,
                'bbox_x': int(bbox_x),
                'bbox_y': int(bbox_y),
                'bbox_z': int(bbox_z),
                'bbox_volume': int(bbox_volume),
                'min_dimension': int(min_dim),
                'density': float(density),
            })
        except Exception as e:
            print(f"    ❌ Error processing {case['case_id']}: {e}")

df = pd.DataFrame(results)

print("\n" + "="*80)
print("Lesion Size Analysis Summary")
print("="*80)

# Overall statistics
print(f"\nTotal cases analyzed: {len(df)}")
print(f"Preprocessed voxel range: {df['preproc_voxels'].min():.0f} - {df['preproc_voxels'].max():.0f}")
print(f"Mean preprocessed voxels: {df['preproc_voxels'].mean():.0f} ± {df['preproc_voxels'].std():.0f}")
print(f"Median preprocessed voxels: {df['preproc_voxels'].median():.0f}")

# Per-dataset statistics
print("\n" + "-"*80)
print("Per-Dataset Breakdown")
print("-"*80)
print(f"{'Dataset':<12} {'Cases':>6} {'Mean Voxels':>12} {'Median Voxels':>14} {'Min Dim':>10}")
print("-"*80)
for dataset in df['dataset'].unique():
    ds_df = df[df['dataset'] == dataset]
    print(f"{dataset:<12} {len(ds_df):>6} {ds_df['preproc_voxels'].mean():>12.0f} "
          f"{ds_df['preproc_voxels'].median():>14.0f} {ds_df['min_dimension'].mean():>10.1f}")

# Thresholding analysis
print("\n" + "-"*80)
print("Filtering Analysis: Cases Remaining by Minimum Voxel Threshold")
print("-"*80)
print(f"{'Threshold':>12} {'Cases':>8} {'Pct':>8}")
print("-"*80)
for threshold in [100, 200, 500, 1000, 2000, 5000]:
    viable = (df['preproc_voxels'] >= threshold).sum()
    pct = viable / len(df) * 100
    print(f"{threshold:>12} {viable:>8} {pct:>7.1f}%")

print("\n" + "-"*80)
print("Filtering Analysis: Cases Remaining by Minimum Dimension Threshold")
print("-"*80)
print(f"{'Threshold':>12} {'Cases':>8} {'Pct':>8}")
print("-"*80)
for threshold in [2, 3, 5, 10, 15, 20]:
    viable = (df['min_dimension'] >= threshold).sum()
    pct = viable / len(df) * 100
    print(f"{threshold:>12} {viable:>8} {pct:>7.1f}%")

# Identify problematic cases
print("\n" + "-"*80)
print("Examples of Very Small Lesions (< 500 voxels or min_dim < 5)")
print("-"*80)
small_cases = df[(df['preproc_voxels'] < 500) | (df['min_dimension'] < 5)].nsmallest(20, 'preproc_voxels')
print(small_cases[['dataset', 'case', 'preproc_voxels', 'bbox_x', 'bbox_y', 'bbox_z', 'min_dimension']].to_string(index=False))

# Save results
output_csv = project_root / 'results' / 'lesion_size_analysis.csv'
output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved detailed results to: {output_csv}")

print("\n" + "="*80)
print("Recommendations")
print("="*80)
print("""
Based on the analysis, consider filtering cases where:

1. Preprocessed voxels < 500: Lesions too small for reliable SAM segmentation
   → Very sparse signal in 128³ volume (< 0.03% of total voxels)

2. Minimum dimension < 5: Lesions too thin in at least one axis
   → Essentially 2D structures that SAM struggles with
   → Example: Liver-001_MR has 2×23×21 bbox (min_dim=2)

3. Density < 0.3: Very sparse/fragmented lesions
   → Hard for SAM to distinguish from noise

Recommended filtering:
  preproc_voxels >= 500 AND min_dimension >= 5 AND density >= 0.3

This will focus evaluation on cases where SAM has a reasonable chance of success.
""")
