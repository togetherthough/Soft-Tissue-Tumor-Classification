"""Create simple aggregate summary CSV of preprocessing impact."""

from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
lesion_csv = project_root / 'results' / 'lesion_size_analysis.csv'

df = pd.read_csv(lesion_csv)

total_cases = len(df)
empty_cases = len(df[df['preproc_voxels'] == 0])
cases_under_500 = len(df[df['preproc_voxels'] < 500])
cases_under_1000 = len(df[df['preproc_voxels'] < 1000])
cases_min_dim_under_5 = len(df[df['min_dimension'] < 5])
cases_min_dim_under_3 = len(df[df['min_dimension'] < 3])
viable_cases = len(df[(df['preproc_voxels'] >= 500) & (df['min_dimension'] >= 5)])

summary = {
    'metric': [
        'total_cases',
        'empty_cases',
        'cases_under_500_voxels',
        'cases_under_1000_voxels',
        'cases_min_dim_under_5',
        'cases_min_dim_under_3',
        'viable_cases_500vox_5dim',
    ],
    'count': [
        total_cases,
        empty_cases,
        cases_under_500,
        cases_under_1000,
        cases_min_dim_under_5,
        cases_min_dim_under_3,
        viable_cases,
    ],
    'percentage': [
        100.0,
        empty_cases / total_cases * 100,
        cases_under_500 / total_cases * 100,
        cases_under_1000 / total_cases * 100,
        cases_min_dim_under_5 / total_cases * 100,
        cases_min_dim_under_3 / total_cases * 100,
        viable_cases / total_cases * 100,
    ]
}

summary_df = pd.DataFrame(summary)
summary_df['percentage'] = summary_df['percentage'].round(1)

output_path = project_root / 'data' / 'preprocessing_summary.csv'
summary_df.to_csv(output_path, index=False)

print("Preprocessing Impact Summary")
print("="*60)
for _, row in summary_df.iterrows():
    print(f"{row['metric']:<30} {row['count']:>5} ({row['percentage']:>5.1f}%)")
print(f"\n✅ Saved to: {output_path}")
