# Pooling Strategy Comparison Guide

## Overview

The pooling strategy comparison experiment now supports selecting specific pooling strategies via the `--pooling-strategies` flag, making it easy to run targeted experiments.

## Available Pooling Strategies

1. **avg** - Global Average Pooling
   - Feature dimension: C (e.g., 384)
   - Fastest to compute
   - Good baseline

2. **multiscale** - Multiscale Pyramid Pooling
   - Scales: 1×1×1, 2×2×2, 4×4×4
   - Feature dimension: C × 73 (e.g., 28,032)
   - Captures multi-scale spatial information
   - Requires PCA due to high dimensionality

3. **percentile** - Percentile Pooling
   - Percentiles: 10th, 25th, 50th, 75th, 90th
   - Feature dimension: C × 5 (e.g., 1,920)
   - Captures distribution of feature activations

## Usage Examples

### Run All Strategies (Default)
```bash
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --output-dir results/pooling_comparison
```

### Run Single Strategy
```bash
# Only average pooling
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --output-dir results/pooling_comparison \
    --pooling-strategies avg

# Only multiscale pooling
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --output-dir results/pooling_comparison \
    --pooling-strategies multiscale
```

### Run Multiple Strategies
```bash
# Compare multiscale and percentile only
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --output-dir results/pooling_comparison \
    --pooling-strategies multiscale percentile
```

### Combine with Dataset Filter
```bash
# Test average pooling on GIST dataset only
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --output-dir results/pooling_comparison \
    --datasets gist \
    --pooling-strategies avg
```

## SLURM Examples

### Quick Test with Single Strategy
```bash
# Modify the SLURM script to test only one strategy
python -u cluster_scripts/experiments/compare_pooling_strategies.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --pooling-strategies avg \
    --roi-margin 30 \
    --n-splits 5
```

### Compare Two Strategies
```bash
python -u cluster_scripts/experiments/compare_pooling_strategies.py \
    --config ${CONFIG_FILE} \
    --output-dir ${RESULTS_DIR} \
    --pooling-strategies multiscale percentile \
    --roi-margin 30 \
    --n-splits 5
```

## All Available Options

```
--config                 : Path to datasets config YAML
--output-dir             : Output directory for results
--datasets               : Specific datasets to run (space-separated)
--pooling-strategies     : Specific pooling strategies (choices: avg, multiscale, percentile)
--roi-margin             : ROI margin in voxels (default: 30)
--img-size               : Image size for SAM-Med3D (default: 128)
--n-splits               : Number of k-fold CV splits (default: 5)
--n-components-max       : Maximum PCA components (default: 500)
--random-state           : Random state for reproducibility (default: 42)
```

## Common Workflows

### 1. Quick Single-Strategy Test
Use when you want to quickly test one strategy on a specific dataset:
```bash
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --output-dir results/test \
    --datasets gist \
    --pooling-strategies avg
```

### 2. Two-Way Comparison
Compare your new strategy against the baseline:
```bash
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --output-dir results/comparison \
    --pooling-strategies avg multiscale
```

### 3. Full Evaluation
Run all strategies on all datasets (original behavior):
```bash
python cluster_scripts/experiments/compare_pooling_strategies.py \
    --config configs/datasets.yaml \
    --output-dir results/full_comparison
```

## Output

Results are saved to:
- `{output_dir}/pooling_comparison_{timestamp}.csv` - Timestamped results
- `{output_dir}/pooling_comparison_latest.csv` - Latest results (overwritten)

The CSV contains:
- `dataset`: Dataset name
- `pooling_strategy`: Strategy used (avg, multiscale, or percentile)
- `accuracy`, `accuracy_std`: Mean and std of accuracy
- `macro_f1`, `macro_f1_std`: Mean and std of macro F1-score
- `roc_auc`, `roc_auc_std`: Mean and std of ROC AUC (if binary)
- `n_samples`: Number of samples
- `n_features_original`, `n_features_pca`: Feature counts

## Tips

1. **Start small**: Test with `--datasets <single_dataset> --pooling-strategies avg` first
2. **Compare incrementally**: Run `avg` first as baseline, then add others
3. **Time considerations**: 
   - `avg` is fastest
   - `multiscale` takes longest (high-dimensional features require PCA)
   - `percentile` is in between
4. **Memory**: Multiscale pooling generates large feature matrices; ensure sufficient GPU memory
