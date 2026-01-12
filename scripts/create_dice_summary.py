"""Create summary statistics from sam_dice_scores_11clicks.csv"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent
dice_csv = project_root / 'results' / 'sam_visualization' / 'sam_dice_scores_11clicks.csv'
output_csv = project_root / 'results' / 'sam_visualization' / 'dice_scores_summary.csv'

# Read the detailed DICE scores
df = pd.read_csv(dice_csv)

# Group by dataset and compute summary statistics
summary_stats = []

for dataset in sorted(df['dataset'].unique()):
    ds_df = df[df['dataset'] == dataset]
    dice_scores = ds_df['dice_score'].dropna()
    
    # Compute 95% confidence interval for the mean
    n = len(dice_scores)
    mean_val = dice_scores.mean()
    std_val = dice_scores.std()
    std_error = std_val / np.sqrt(n)
    
    # Use t-distribution for 95% CI
    confidence_level = 0.95
    degrees_freedom = n - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_error = t_critical * std_error
    
    ci_lower = mean_val - margin_error
    ci_upper = mean_val + margin_error
    
    summary_stats.append({
        'dataset': dataset,
        'n_cases': len(ds_df),
        'mean_dice': mean_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_dice': std_val,
        'median_dice': dice_scores.median(),
        'min_dice': dice_scores.min(),
        'max_dice': dice_scores.max(),
        'mean_gt_voxels': ds_df['gt_voxels'].mean(),
        'mean_pred_voxels': ds_df['pred_voxels'].mean(),
    })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_stats)

# Round to 4 decimal places
numeric_cols = ['mean_dice', 'ci_lower', 'ci_upper', 'std_dice', 'median_dice', 'min_dice', 'max_dice', 
                'mean_gt_voxels', 'mean_pred_voxels']
for col in numeric_cols:
    summary_df[col] = summary_df[col].round(4)

# Save
summary_df.to_csv(output_csv, index=False)

print("DICE Score Summary by Dataset")
print("="*80)
print(summary_df.to_string(index=False))
print(f"\nSummary saved to: {output_csv}")
