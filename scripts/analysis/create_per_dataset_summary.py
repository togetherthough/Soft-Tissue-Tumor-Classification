"""Create per-dataset breakdown of preprocessing impact."""

from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
lesion_csv = project_root / 'results' / 'lesion_size_analysis.csv'

df = pd.read_csv(lesion_csv)

# Overall summary first
total_cases = len(df)
empty_cases = len(df[df['preproc_voxels'] == 0])
cases_under_500 = len(df[df['preproc_voxels'] < 500])
cases_under_1000 = len(df[df['preproc_voxels'] < 1000])
cases_min_dim_under_5 = len(df[df['min_dimension'] < 5])
cases_min_dim_under_3 = len(df[df['min_dimension'] < 3])
viable_cases = len(df[(df['preproc_voxels'] >= 500) & (df['min_dimension'] >= 5)])

overall = {
    'dataset': 'OVERALL',
    'total_cases': total_cases,
    'empty_cases': empty_cases,
    'empty_pct': empty_cases / total_cases * 100,
    'cases_under_500_voxels': cases_under_500,
    'under_500_pct': cases_under_500 / total_cases * 100,
    'cases_under_1000_voxels': cases_under_1000,
    'under_1000_pct': cases_under_1000 / total_cases * 100,
    'cases_min_dim_under_5': cases_min_dim_under_5,
    'min_dim_under_5_pct': cases_min_dim_under_5 / total_cases * 100,
    'cases_min_dim_under_3': cases_min_dim_under_3,
    'min_dim_under_3_pct': cases_min_dim_under_3 / total_cases * 100,
    'viable_cases': viable_cases,
    'viable_pct': viable_cases / total_cases * 100,
}

# Per-dataset breakdown
results = [overall]

for dataset in sorted(df['dataset'].unique()):
    ds_df = df[df['dataset'] == dataset]
    
    total = len(ds_df)
    empty = len(ds_df[ds_df['preproc_voxels'] == 0])
    under_500 = len(ds_df[ds_df['preproc_voxels'] < 500])
    under_1000 = len(ds_df[ds_df['preproc_voxels'] < 1000])
    min_dim_under_5 = len(ds_df[ds_df['min_dimension'] < 5])
    min_dim_under_3 = len(ds_df[ds_df['min_dimension'] < 3])
    viable = len(ds_df[(ds_df['preproc_voxels'] >= 500) & (ds_df['min_dimension'] >= 5)])
    
    results.append({
        'dataset': dataset,
        'total_cases': total,
        'empty_cases': empty,
        'empty_pct': empty / total * 100 if total > 0 else 0,
        'cases_under_500_voxels': under_500,
        'under_500_pct': under_500 / total * 100 if total > 0 else 0,
        'cases_under_1000_voxels': under_1000,
        'under_1000_pct': under_1000 / total * 100 if total > 0 else 0,
        'cases_min_dim_under_5': min_dim_under_5,
        'min_dim_under_5_pct': min_dim_under_5 / total * 100 if total > 0 else 0,
        'cases_min_dim_under_3': min_dim_under_3,
        'min_dim_under_3_pct': min_dim_under_3 / total * 100 if total > 0 else 0,
        'viable_cases': viable,
        'viable_pct': viable / total * 100 if total > 0 else 0,
    })

summary_df = pd.DataFrame(results)

# Round percentages
pct_cols = [col for col in summary_df.columns if '_pct' in col]
for col in pct_cols:
    summary_df[col] = summary_df[col].round(1)

output_path = project_root / 'data' / 'preprocessing_summary_per_dataset.csv'
summary_df.to_csv(output_path, index=False)

print("Preprocessing Impact - Per Dataset Breakdown")
print("="*80)
print(f"\n{'Dataset':<12} {'Total':>6} {'Empty':>6} {'<500vox':>8} {'<1000vox':>9} {'min_dim<5':>10} {'Viable':>8}")
print("-"*80)
for _, row in summary_df.iterrows():
    print(f"{row['dataset']:<12} {row['total_cases']:>6} "
          f"{row['empty_cases']:>6} ({row['empty_pct']:>4.1f}%) "
          f"{row['cases_under_500_voxels']:>6} ({row['under_500_pct']:>4.1f}%) "
          f"{row['cases_under_1000_voxels']:>6} ({row['under_1000_pct']:>4.1f}%) "
          f"{row['cases_min_dim_under_5']:>6} ({row['min_dim_under_5_pct']:>4.1f}%) "
          f"{row['viable_cases']:>6} ({row['viable_pct']:>4.1f}%)")

print(f"\n✅ Saved to: {output_path}")
